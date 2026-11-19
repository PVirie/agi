import torch
import torch.nn as nn
import numpy as np

from interfaces.network import Actor_Network, Policy_Value_Network
from implementations.networks.torch.components.base import Categorical_With_Mask, init_weights
from implementations.networks.torch.components.std_conv import FiLMMiniGridCNN
from implementations.networks.torch.components.std_lstm import Sample_Type, Straight_Through_Recurrent_Cell
from implementations.networks.torch.components.transformer import InstructionTransformer, get_padding_mask
from implementations.networks.torch.policy.base import Policy_Core as Base_Policy_Core
from utilities.safe_torch_module import Safe_nn_Module


class Policy_Core(Base_Policy_Core, Actor_Network, Policy_Value_Network):

    def __init__(self, 
                 int_action_size, ext_action_size,
                 node_dim,
                 internal_state_size,
                 goal_size,
                 inventory_size,
                 width, height, channel,
                 dict_size, embedding_dim, pad_token_id,
                 hidden_size, layers, 
                 cnn_channels=128, cnn_blocks=2,
                 value_size=1,
                 cell="lstm",
                 device=None, 
                 persistence_path=None, first_load_path=None):
        """
        Pipeline per step:
            instruction, inventory -> embedding --------------------+
            graph context (head + edge nodes) -> node projection ---+-> transformer -> CLS --+-> linear -> latent
            obs -> FiLM CNN (conditioned on the CLS) ----------------------------------------+
            latent (+ prev ext action), prev memory state, prev internal output
                -> Straight_Through_Recurrent_Cell -> internal output (int flag, edge_1, edge_2, write value), memory state, hidden
            hidden -> ext head, value head
        Training signals:
            ext head: RL; log-prob and entropy are the ext action's only
            internal heads: backpropagation only, straight-through the sampled output fed into later steps,
            plus their entropy as the aux loss (get_log_prob_with_value_aux, weighted by the learner's aux_coef);
            int flag and edges are categorical (fed back as one-hots), the graph write value is continuous in [0, 1]^node_dim
        Graph nodes are transformer tokens after the instruction and inventory (projected to embedding_dim if node_dim differs):
        they are never padding, and the positional encoding tells the slots apart
        (slot 0 is the head node, slot i the i-th edge, which the edge heads index).
        """
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="babyai_lstm_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.int_action_size = int_action_size  # num classes for flag
        self.ext_action_size = ext_action_size
        self.internal_state_size = internal_state_size
        self.edge_size = internal_state_size - 1  # edge heads pick one of the head node's edge slots
        self.goal_size = goal_size
        self.inventory_size = inventory_size
        self.width = width
        self.height = height
        self.channel = channel
        self.dict_size = dict_size
        self.embedding_dim = embedding_dim
        self.node_dim = node_dim
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        # 1 for single-stream PPO (values squeezed to (batch, context)); 2 for ppo_graph_dual [extrinsic, intrinsic]
        self.value_size = value_size

        self.position_size = 2
        self.content_size = internal_state_size * node_dim + goal_size + inventory_size + width * height * channel
        # one recurrent layer per entry of layers, each with a hidden_size state; the stacked state is the memory
        # cell: "lstm" (state = unbounded cell states c), "gru" (state = hidden states h in [-1, 1])
        # or "residual" (state = unbounded residual states s, s' = s + update)
        self.cell = cell
        self.memory_size = len(layers) * hidden_size
        # the internal outputs [int flag, edge_1, edge_2, write value (node_dim)] are contiguous, in the recurrent cell's head order,
        # in both the action and the context; ext comes before them
        self.internal_output_size = 1 + self.position_size + node_dim
        # action: [ext, internal outputs, memory stack (not sampled)]
        self.packed_action_size = 1 + self.internal_output_size + self.memory_size
        # context: [reward, ext, internal outputs, memory reset, memory stack, content]
        # the previous internal outputs and memory stack (+ a flag marking where it restarts) ride in the context:
        # acting reads them per step, learning slides the recurrence through time and only restarts from them at flagged steps
        self.packed_context_size = 1 + 1 + self.internal_output_size + 1 + self.memory_size + self.content_size

        self.ext_pos = 1
        self.internal_pos = self.ext_pos + 1
        self.position_pos = self.internal_pos + 1
        self.write_pos = self.position_pos + self.position_size
        self.memory_reset_pos = self.write_pos + node_dim
        self.memory_pos = self.memory_reset_pos + 1
        self.internal_state_pos = self.memory_pos + self.memory_size
        self.goal_pos = self.internal_state_pos + self.internal_state_size * node_dim
        self.inv_pos = self.goal_pos + goal_size
        self.obs_pos = self.inv_pos + inventory_size

        self.goal_embedding = nn.Embedding(dict_size, embedding_dim, padding_idx=pad_token_id)  # for goal tokens
        self.inv_embedding = nn.Embedding(dict_size, embedding_dim, padding_idx=pad_token_id)  # for inventory tokens
        self.node_projection = nn.Linear(node_dim, embedding_dim) if node_dim != embedding_dim else nn.Identity()

        self.goal_feature_extraction = InstructionTransformer(
            input_dim=embedding_dim,
            d_model=hidden_size,
            nhead=8,
            num_layers=4,
            dropout=0.01,
            max_len=goal_size + inventory_size + internal_state_size,
            use_pe=True,  # mission semantics depend on word order (attribute binding, "then"/"after")
            pooling="cls",  # mean pooling dilutes order-dependent structure ("A then B" vs "B then A")
        )

        # vision conditioned via FiLM on the graph-aware instruction features
        self.conv_layers = FiLMMiniGridCNN(
            output_dims=hidden_size, 
            cond_dims=hidden_size,
            input_channels=channel, 
            width=width, height=height,
            vocab_size=256, embedding_dim=8,
            channels=cnn_channels, num_blocks=cnn_blocks
        )

        self.latent_projection = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # internal heads [int flag, edge_1, edge_2, write value] live in the cell: their samples feed back into the recurrence.
        # the previous ext action is a plain input: its effect goes through the environment, which is not differentiable
        self.recurrent_cell = Straight_Through_Recurrent_Cell(
            output_dims=hidden_size,
            input_dims=hidden_size + ext_action_size,
            hidden_dims=hidden_size,
            layers=[hidden_size for _ in layers],
            heads=[
                (Sample_Type.CATEGORICAL, int_action_size),
                (Sample_Type.CATEGORICAL, self.edge_size),
                (Sample_Type.CATEGORICAL, self.edge_size),
                (Sample_Type.CONTINUOUS, node_dim),
            ],
            cell=cell,
        )

        self.head_ext = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, ext_action_size)
        )

        self.head_value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, value_size)
        )

        self.reset_parameters()
        self.load(override_persistence_path=first_load_path)
        self.eval()


    def reset_parameters(self):
        # Reset parameters of all layers
        self.goal_embedding.reset_parameters()
        self.inv_embedding.reset_parameters()
        self.node_projection.apply(init_weights)

        self.goal_feature_extraction.reset_parameters()
        self.conv_layers.reset_parameters()
        self.latent_projection.apply(init_weights)
        self.recurrent_cell.reset_parameters()

        def init_actor_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.head_ext.apply(init_actor_weights)
        # internal heads [int flag, edge_1, edge_2, write value] inside the recurrent cell
        for head in self.recurrent_cell.heads:
            head.apply(init_actor_weights)

        def init_value_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        self.head_value.apply(init_value_weights)


    def encode(self, internal_state, goal, inv, obs):
        # internal_state: (batch, context_size, internal_state_size * node_dim); goal, inv: token ids; obs: flattened grid
        # returns the fused latent (batch, context_size, hidden_size); every step is independent, so time is batched
        batch_size, context_size = internal_state.shape[:2]
        n = batch_size * context_size

        node_tokens = self.node_projection(
            internal_state.reshape(batch_size, context_size, self.internal_state_size, self.node_dim)
        )  # (batch_size, context_size, internal_state_size, embedding_dim)

        padding_mask = torch.concat([
            get_padding_mask(torch.concat([goal, inv], dim=-1), self.pad_token_id),
            node_tokens.new_ones(batch_size, context_size, self.internal_state_size)
        ], dim=-1)  # (batch_size, context_size, goal_size + inventory_size + internal_state_size)
        tokens = torch.concat([
            self.goal_embedding(goal.long()),
            self.inv_embedding(inv.long()),
            node_tokens
        ], dim=2)  # (batch_size, context_size, goal_size + inventory_size + internal_state_size, embedding_dim)
        instruction_summary = self.goal_feature_extraction(tokens, padding_mask)  # (batch_size, context_size, hidden_size)

        obs = torch.reshape(obs, (n, self.height, self.width, self.channel))
        obs_summary = self.conv_layers(obs, instruction_summary.reshape(n, self.hidden_size))  # (n, hidden_size)

        latent = self.latent_projection(torch.concat([instruction_summary.reshape(n, self.hidden_size), obs_summary], dim=-1))
        return latent.view(batch_size, context_size, self.hidden_size)


    def compute(self, context, available_flags=None, sample=False):
        # context has shape (batch, context_size, self.packed_context_size); available_flags: int-flag valid mask or None
        # sample: also draw the internal actions at every step; the ext action is sampled from the returned ext logits
        prev_ext = context[:, :, self.ext_pos].long()  # (batch_size, context_size)
        prev_internal = context[:, :, self.internal_pos: self.memory_reset_pos]  # (batch_size, context_size, 3 + node_dim) [int flag, edge_1, edge_2, write value]
        memory_reset = context[:, :, self.memory_reset_pos] > 0.5  # (batch_size, context_size)
        prev_memory_state = context[:, :, self.memory_pos: self.internal_state_pos]  # (batch_size, context_size, memory_size)
        internal_state = context[:, :, self.internal_state_pos: self.goal_pos]  # (batch_size, context_size, internal_state_size * node_dim)
        goal = context[:, :, self.goal_pos: self.inv_pos]  # (batch_size, context_size, goal_size)
        inv = context[:, :, self.inv_pos: self.obs_pos]  # (batch_size, context_size, inventory_size)
        obs = context[:, :, self.obs_pos:]  # (batch_size, context_size, obs_size)

        latent = self.encode(internal_state, goal, inv, obs)  # (batch_size, context_size, hidden_size)

        prev_ext_one_hot = nn.functional.one_hot(prev_ext, num_classes=self.ext_action_size).float()

        internal_params, memory_state, hidden, internal_samples = self.recurrent_cell(
            torch.concat([latent, prev_ext_one_hot], dim=-1), prev_internal, prev_memory_state, memory_reset,
            masks=[available_flags, None, None, None], sample=sample
        )  # [(batch_size, context_size, param_size)] * 4, (batch_size, context_size, memory_size), (batch_size, context_size, hidden_size), (batch_size, context_size, 3 + node_dim) or None

        ext_logits = self.head_ext(hidden)  # (batch_size, context_size, ext_action_size)

        values = self.head_value(hidden)  # (batch_size, context_size, value_size)
        if self.value_size == 1:
            values = values.squeeze(-1)  # (batch_size, context_size)

        return ext_logits, internal_params, memory_state, values, internal_samples


    def _split_valid_actions(self, valid_actions):
        if valid_actions is None:
            return None, None
        if isinstance(valid_actions, np.ndarray):
            valid_actions = torch.tensor(valid_actions, dtype=torch.bool)
        valid_actions = valid_actions.to(self.device)
        # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
        return valid_actions[:, :, :self.int_action_size], valid_actions[:, :, self.int_action_size:]


    def get_action(self, context, valid_actions=None):

        if isinstance(context, np.ndarray):
            # float: node values are continuous and would be truncated by a long cast
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        available_flags, available_actions = self._split_valid_actions(valid_actions)

        with torch.no_grad():
            ext_logits, _, memory_state, _, internal_samples = self.compute(context, available_flags, sample=True)
            ext_action = Categorical_With_Mask(logits=ext_logits, mask=available_actions).sample()

        action = torch.concat([ext_action.unsqueeze(-1).float(), internal_samples, memory_state], dim=-1)
        return action.cpu().numpy().astype(float)
    

    def get_log_prob_with_value_aux(self, context, selected_action, valid_actions=None):
        log_probs, entropy, values, internal_entropy = self._evaluate(context, selected_action, valid_actions)
        # aux loss (batch, context_size): minimizing it maximizes the internal heads' entropy, keeping their softmax from collapsing
        return log_probs, entropy, values, -internal_entropy


    def get_log_probability_with_value(self, context, selected_action, valid_actions=None):
        return self._evaluate(context, selected_action, valid_actions)[:3]


    def _evaluate(self, context, selected_action, valid_actions=None):
        # now context has shape (batch, context_size + 1, self.packed_context_size)
        # returns the ext action's log-prob and entropy (batch, context_size), values (batch, context_size + 1),
        # and the internal heads' entropy summed over heads (batch, context_size).
        # the internal heads have no log-prob: they are trained by the gradient of these terms (and the value loss)
        # flowing back straight-through the sampled internal outputs that the recurrence feeds into later steps.
        # the write value's entropy is averaged over its node_dim dims, so it weighs as one head

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)
        if isinstance(selected_action, np.ndarray):
            selected_action = torch.tensor(selected_action, dtype=torch.float32).to(self.device)

        available_flags, available_actions = self._split_valid_actions(valid_actions)

        ext_logits, internal_params, _, values, _ = self.compute(context, available_flags)
        # values keep all contexts: the last one is the bootstrap value
        # drop the last context for everything else: it is the observation after the last action
        probs_ext = Categorical_With_Mask(logits=ext_logits[:, :-1, :], mask=available_actions)
        internal_entropy = self.recurrent_cell.entropy(
            [p[:, :-1, :] for p in internal_params], [available_flags, None, None, None]
        )
        return probs_ext.log_prob(selected_action[:, :, 0]), probs_ext.entropy(), values, internal_entropy


    def unpack_action(self, packed_action):
        # packed_action (batch, packed_action_size): [ext, int flag, edge_1, edge_2, write value, memory stack]
        # returns int flag (batch,), ext action (batch, 1), position (edge_1, edge_2) (batch, 2),
        # and the sampled graph write value (batch, node_dim) in [0, 1]; see unpack_memory for the memory stack
        internal = packed_action[:, 1: 1 + self.internal_output_size]
        int_part = internal[:, 0].astype(int)
        ext_part = packed_action[:, 0:1].astype(int)
        position = internal[:, 1: 1 + self.position_size].astype(float)
        write_value = internal[:, 1 + self.position_size:].astype(float)
        return int_part, ext_part, position, write_value


    def unpack_memory(self, packed_action):
        # (batch, memory_size): every recurrent layer's state, bottom layer first
        return packed_action[:, 1 + self.internal_output_size:].astype(float)


    def pack_context(self, b_reward=None, b_int=None, b_ext=None, b_position=None, b_write=None, b_memory_reset=None, b_memory=None, b_content=None):
        # b_write: previous write value (batch, node_dim), see unpack_action; fed back into the recurrence like the other internal outputs
        # b_memory: previous memory stack (batch, memory_size), see unpack_memory
        # b_memory_reset: (batch,) bool, True where the recurrence restarts from b_memory (e.g. an episode start with zeros)
        batch_size = next(b.shape[0] for b in (b_reward, b_int, b_ext, b_position, b_write, b_memory_reset, b_memory, b_content) if b is not None)
        if b_reward is None:
            b_reward = np.zeros((batch_size,), dtype=float)
        if b_int is None:
            b_int = np.zeros((batch_size,), dtype=int)
        if b_ext is None:
            b_ext = np.zeros((batch_size, 1), dtype=int)
        if b_position is None:
            b_position = np.zeros((batch_size, self.position_size), dtype=float)
        if b_content is None:
            b_content = np.zeros((batch_size, self.content_size), dtype=float)
        if b_write is None:
            b_write = np.zeros((batch_size, self.node_dim), dtype=float)
        if b_memory is None:
            b_memory = np.zeros((batch_size, self.memory_size), dtype=float)
            b_memory_reset = np.ones((batch_size,), dtype=bool)
        if b_memory_reset is None:
            b_memory_reset = np.zeros((batch_size,), dtype=bool)
        return np.concatenate([
            np.reshape(b_reward, (batch_size, 1)),
            np.reshape(b_ext, (batch_size, 1)),
            np.reshape(b_int, (batch_size, 1)),
            np.reshape(b_position, (batch_size, self.position_size)),
            np.reshape(b_write, (batch_size, self.node_dim)),
            np.reshape(b_memory_reset, (batch_size, 1)),
            np.reshape(b_memory, (batch_size, self.memory_size)),
            np.reshape(b_content, (batch_size, self.content_size))
        ], axis=-1).astype(float)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    C, B, T, D = 3, 4, 5, 12  # node_dim D != embedding_dim (projected)
    core = Policy_Core(
        int_action_size=6, ext_action_size=8,
        node_dim=D, internal_state_size=1 + C,
        goal_size=5, inventory_size=3,
        width=7, height=7, channel=3,
        dict_size=32, embedding_dim=8, pad_token_id=0,
        hidden_size=32, layers=[1, 1],
        cnn_channels=16, cnn_blocks=1,
        cell="gru",
        device=torch.device("cpu"),
    )

    def random_content():
        return np.concatenate([
            np.random.uniform(-1, 1, (B, (1 + C) * D)),
            np.random.randint(0, 32, (B, 5 + 3)),
            np.random.randint(0, 10, (B, 7 * 7 * 3)),
        ], axis=-1)

    valid = np.random.rand(B, T + 1, 6 + 8) > 0.3
    valid[:, :, 0] = True
    valid[:, :, 6] = True

    # every graph slot is a transformer token, so changing any one node changes the latent
    def encode_context(context):
        return core.encode(
            context[:, :, core.internal_state_pos: core.goal_pos], context[:, :, core.goal_pos: core.inv_pos],
            context[:, :, core.inv_pos: core.obs_pos], context[:, :, core.obs_pos:]
        )

    with torch.no_grad():
        probe = torch.tensor(core.pack_context(b_reward=np.zeros(B), b_int=np.zeros(B, dtype=int), b_ext=np.zeros((B, 1), dtype=int),
                                               b_position=np.zeros((B, 2)), b_content=random_content()), dtype=torch.float32)[:, None]
        for slot in range(1 + C):
            perturbed = probe.clone()
            perturbed[:, :, core.internal_state_pos + slot * D] += 1.0
            assert not torch.allclose(encode_context(probe), encode_context(perturbed), atol=1e-6), f"graph slot {slot} does not reach the latent"

    # act one step at a time as model_81 does: the context carries the executed action and the memory
    contexts, actions = [], []
    last_action = np.zeros((B, core.packed_action_size))
    for t in range(T + 1):
        last_int, last_ext, last_position, last_write = core.unpack_action(last_action)
        context = core.pack_context(
            b_reward=np.zeros(B),
            b_int=last_int,
            b_ext=last_ext,
            b_position=last_position,
            b_write=last_write,
            b_memory_reset=np.full(B, t == 0),
            b_memory=core.unpack_memory(last_action),
            b_content=random_content(),
        )
        last_action = core.get_action(context[:, None], valid[:, t:t + 1])[:, -1]
        contexts.append(context)
        actions.append(last_action)
    contexts = torch.tensor(np.stack(contexts, axis=1), dtype=torch.float32)
    actions = torch.tensor(np.stack(actions, axis=1), dtype=torch.float32)
    valid = torch.tensor(valid)
    assert (valid[..., :6].gather(-1, actions[..., 1:2].long()).all() and valid[..., 6:].gather(-1, actions[..., 0:1].long()).all()), "sampled actions must be valid"

    # learning slides through time with straight-through feedback, yet reproduces acting exactly
    flags = core._split_valid_actions(valid[:, :T])[0]
    ext_logits, internal_logits, memory_state, values, _ = core.compute(contexts, flags)
    assert torch.allclose(memory_state, actions[:, :, 4 + D:], atol=1e-5), "unrolled memory differs from the acted memory"
    assert np.allclose(core.unpack_memory(actions[:, -1].numpy()), actions[:, -1, 4 + D:].numpy())
    write_value = core.unpack_action(actions[:, -1].numpy())[3]
    assert core.memory_size == 2 * 32 and write_value.shape == (B, D), "the memory stack is hidden_size per layer, the write value node_dim"
    assert ((write_value >= 0) & (write_value <= 1)).all() and np.unique(write_value).size > 1, "the write value is sampled in [0, 1]^node_dim"
    assert internal_logits[3].shape == (B, T + 1, 2 * D), "the write head outputs one Beta (alpha, beta) per dim"
    with torch.no_grad():
        for t in range(T + 1):
            step_ext, step_internal = core.compute(contexts[:, t:t + 1])[:2]
            for head, step_head in zip([ext_logits] + internal_logits, [step_ext] + step_internal):
                assert torch.allclose(head[:, t], step_head[:, 0], atol=1e-5), f"unrolled logits differ from acting at step {t}"
    assert values.shape == (B, T + 1)

    # RL sees the ext action only; the internal heads' entropy is the aux loss
    log_probs, entropy, values, aux = core.get_log_prob_with_value_aux(contexts, actions[:, :T], valid[:, :T])
    assert log_probs.shape == (B, T) and entropy.shape == (B, T) and aux.shape == (B, T)
    assert torch.isfinite(log_probs).all(), "acted ext actions must have finite log-probs"
    assert torch.isfinite(aux).all() and (aux < 0).all(), "aux must be the negative internal entropy"
    assert torch.equal(core.get_log_probability_with_value(contexts, actions[:, :T], valid[:, :T])[0], log_probs)

    # a loss on step 1's ext log-prob reaches the internal heads only straight-through step 0's sampled output
    def internal_head_grads(contexts):
        core.zero_grad()
        log_probs = core.get_log_probability_with_value(contexts, actions[:, :T], valid[:, :T])[0]
        log_probs[:, 1].sum().backward()
        return [torch.cat([p.grad.flatten() for p in head.parameters()]) for head in core.recurrent_cell.heads]

    assert all(g.abs().sum() > 0 for g in internal_head_grads(contexts)), "internal heads must be trained by later steps' losses"
    restarted = contexts.clone()
    restarted[:, 1, core.memory_reset_pos] = 1.0
    assert all(g.abs().sum() == 0 for g in internal_head_grads(restarted)), "a restart must cut the internal-output gradient path"

    # the gradient is evaluated at the output actually taken: replaying a different step 0 flag or write value changes it
    other_flag = contexts.clone()
    other_flag[:, 1, core.internal_pos] = (other_flag[:, 1, core.internal_pos] + 1) % 6
    assert not torch.allclose(internal_head_grads(contexts)[0], internal_head_grads(other_flag)[0]), "the gradient must depend on the taken output"
    other_write = contexts.clone()
    other_write[:, 1, core.write_pos: core.write_pos + D] = 1 - other_write[:, 1, core.write_pos: core.write_pos + D]
    assert not torch.allclose(internal_head_grads(contexts)[3], internal_head_grads(other_write)[3]), "the write gradient must depend on the taken value"

    print("All tests passed!")
