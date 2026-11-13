import torch
import torch.nn as nn
import numpy as np

from interfaces.network import Actor_Network, Policy_Value_Network
from implementations.networks.torch.components.base import Categorical_With_Mask
from implementations.networks.torch.components.std_resnet import ResNet
from implementations.networks.torch.components.std_conv import FiLMMiniGridCNN
from implementations.networks.torch.components.std_lstm import Soft_Feedback_Recurrent_Cell
from implementations.networks.torch.components.transformer import InstructionTransformer, Cross_Modal_Fusion, get_padding_mask
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
                 fusion_queries=4, fusion_layers=1,
                 device=None, 
                 persistence_path=None, first_load_path=None):
        """
        Pipeline per step:
            graph context (head + edge nodes) -> MLP ------------------------------+
            instruction, inventory -> embedding -> transformer (graph cond token) --+-> cross-modal fusion -> latent
            obs -> FiLM CNN (conditioned on the instruction CLS) -------------------+
            latent (+ prev ext action), prev recurrent state, prev internal probabilities
                -> Soft_Feedback_Recurrent_Cell -> internal logits (int flag, edge_1, edge_2), recurrent state, internal probabilities, hidden
            hidden -> ext head, value head
        Training signals:
            ext head: RL; log-prob and entropy are the ext action's only
            internal heads: backpropagation only, through their probabilities fed into later steps
            (the sampled internal actions drive the graph memory but are not fed back),
            plus their entropy as the aux loss (get_log_prob_with_value_aux, weighted by the learner's aux_coef)
            write value: the top recurrent layer's state, trained by backpropagation through time
        fusion_queries, fusion_layers: learned queries and cross-attention layers of the fusion read
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
        # memory = [internal head probabilities fed back to the next step, recurrent state stack]
        # one recurrent layer per entry of layers, each with a node_dim state
        # cell: "lstm" (state = unbounded cell states c) or "gru" (state = hidden states h in [-1, 1])
        self.cell = cell
        self.feedback_size = int_action_size + 2 * self.edge_size
        self.memory_size = self.feedback_size + len(layers) * node_dim
        # the action carries the whole memory (not sampled); its last node_dim entries (the top layer) are the graph write value
        self.packed_action_size = 1 + 1 + self.position_size + self.memory_size
        # previous memory (+ a flag marking where it restarts) rides in the context:
        # acting reads it per step, learning slides the recurrence through time and only restarts from it at flagged steps
        self.packed_context_size = 1 + 1 + 1 + self.position_size + 1 + self.memory_size + self.content_size

        self.position_pos = 1 + 1 + 1
        self.memory_reset_pos = self.position_pos + self.position_size
        self.memory_pos = self.memory_reset_pos + 1
        self.internal_state_pos = self.memory_pos + self.memory_size
        self.goal_pos = self.internal_state_pos + self.internal_state_size * node_dim
        self.inv_pos = self.goal_pos + goal_size
        self.obs_pos = self.inv_pos + inventory_size

        self.goal_embedding = nn.Embedding(dict_size, embedding_dim, padding_idx=pad_token_id)  # for goal tokens
        self.inv_embedding = nn.Embedding(dict_size, embedding_dim, padding_idx=pad_token_id)  # for inventory tokens

        self.goal_feature_extraction = InstructionTransformer(
            input_dim=embedding_dim,
            d_model=hidden_size,
            nhead=8,
            num_layers=2,
            dropout=0.01,
            max_len=goal_size + inventory_size,
            use_pe=True,  # mission semantics depend on word order (attribute binding, "then"/"after")
            pooling="cls",  # mean pooling dilutes order-dependent structure ("A then B" vs "B then A")
            cond_dim=hidden_size,  # graph context token lets memory select which mission clause to attend
        )

        # graph summary over the flattened context, so it keeps which slot holds which node
        self.internal_state_feature_extraction = ResNet(
            output_dims=hidden_size,
            input_dims=internal_state_size * node_dim,
            hidden_dims=hidden_size,
            layers=[hidden_size for _ in layers]
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

        self.fusion = Cross_Modal_Fusion(
            hidden_size=hidden_size,
            node_dim=node_dim,
            num_nodes=internal_state_size,
            map_channels=cnn_channels,
            num_cells=width * height,
            num_queries=fusion_queries,
            num_layers=fusion_layers,
        )

        # internal heads [int flag, edge_1, edge_2] live in the cell: their probabilities feed back into the recurrence.
        # the previous ext action is a plain input: its effect goes through the environment, which is not differentiable
        self.recurrent_cell = Soft_Feedback_Recurrent_Cell(
            output_dims=hidden_size,
            input_dims=hidden_size + ext_action_size,
            hidden_dims=node_dim,
            layers=[hidden_size for _ in layers],
            head_sizes=[int_action_size, self.edge_size, self.edge_size],
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

        self.goal_feature_extraction.reset_parameters()
        self.internal_state_feature_extraction.reset_parameters()
        self.conv_layers.reset_parameters()
        self.fusion.reset_parameters()
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
        # internal heads [int flag, edge_1, edge_2] inside the recurrent cell
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

        graph_summary = self.internal_state_feature_extraction(internal_state)  # (batch_size, context_size, hidden_size)

        goal_inv = torch.concat([goal, inv], dim=-1)  # (batch_size, context_size, goal_size + inventory_size)
        padding_mask = get_padding_mask(goal_inv, self.pad_token_id)  # (batch_size, context_size, goal_size + inventory_size)
        goal_inv_embedded = torch.concat([
            self.goal_embedding(goal.long()),
            self.inv_embedding(inv.long())
        ], dim=2)  # (batch_size, context_size, goal_size + inventory_size, embedding_dim)
        instruction_summary, instruction_tokens = self.goal_feature_extraction(
            goal_inv_embedded, padding_mask, cond=graph_summary, return_tokens=True
        )  # (batch_size, context_size, hidden_size), (batch_size, context_size, goal_size + inventory_size, hidden_size)

        obs = torch.reshape(obs, (n, self.height, self.width, self.channel))
        obs_summary, obs_map = self.conv_layers(obs, instruction_summary.reshape(n, self.hidden_size), return_map=True)

        latent = self.fusion(
            graph_nodes=internal_state.reshape(n, self.internal_state_size, self.node_dim),
            graph_summary=graph_summary.reshape(n, self.hidden_size),
            instruction_tokens=instruction_tokens.reshape(n, -1, self.hidden_size),
            instruction_mask=padding_mask.reshape(n, -1),
            instruction_summary=instruction_summary.reshape(n, self.hidden_size),
            obs_map=obs_map,
            obs_summary=obs_summary,
        )
        return latent.view(batch_size, context_size, self.hidden_size)


    def compute(self, context, available_flags=None, sample=False):
        # context has shape (batch, context_size, self.packed_context_size); available_flags: int-flag valid mask or None
        # sample: also draw the internal actions at every step; the ext action is sampled from the returned ext logits
        prev_ext = context[:, :, 2].long()  # (batch_size, context_size)
        memory_reset = context[:, :, self.memory_reset_pos] > 0.5  # (batch_size, context_size)
        prev_memory = context[:, :, self.memory_pos: self.internal_state_pos]  # (batch_size, context_size, memory_size)
        prev_feedback = prev_memory[:, :, :self.feedback_size]  # (batch_size, context_size, feedback_size)
        prev_state = prev_memory[:, :, self.feedback_size:]  # (batch_size, context_size, len(layers) * node_dim)
        internal_state = context[:, :, self.internal_state_pos: self.goal_pos]  # (batch_size, context_size, internal_state_size * node_dim)
        goal = context[:, :, self.goal_pos: self.inv_pos]  # (batch_size, context_size, goal_size)
        inv = context[:, :, self.inv_pos: self.obs_pos]  # (batch_size, context_size, inventory_size)
        obs = context[:, :, self.obs_pos:]  # (batch_size, context_size, obs_size)

        latent = self.encode(internal_state, goal, inv, obs)  # (batch_size, context_size, hidden_size)

        prev_ext_one_hot = nn.functional.one_hot(prev_ext, num_classes=self.ext_action_size).float()

        internal_logits, state, feedback, hidden, internal_samples = self.recurrent_cell(
            torch.concat([latent, prev_ext_one_hot], dim=-1), prev_feedback, prev_state, memory_reset,
            masks=[available_flags, None, None], sample=sample
        )  # [(batch_size, context_size, size)] * 3, (batch_size, context_size, len(layers) * node_dim), (batch_size, context_size, feedback_size), (batch_size, context_size, hidden_size), (batch_size, context_size, 3) or None
        memory_state = torch.concat([feedback, state], dim=-1)  # (batch_size, context_size, memory_size)

        ext_logits = self.head_ext(hidden)  # (batch_size, context_size, ext_action_size)

        values = self.head_value(hidden)  # (batch_size, context_size, value_size)
        if self.value_size == 1:
            values = values.squeeze(-1)  # (batch_size, context_size)

        return ext_logits, internal_logits, memory_state, values, internal_samples


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

        action = np.concatenate([
            internal_samples[:, :, 0:1].cpu().numpy(),  # int flag
            ext_action.unsqueeze(-1).cpu().numpy(),
            internal_samples[:, :, 1:3].cpu().numpy(),  # edge_1, edge_2
            memory_state.cpu().numpy()
        ], axis=-1)

        return action.astype(float)
    

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
        # flowing back through the internal probabilities that the recurrence feeds into later steps

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)
        if isinstance(selected_action, np.ndarray):
            selected_action = torch.tensor(selected_action, dtype=torch.float32).to(self.device)

        available_flags, available_actions = self._split_valid_actions(valid_actions)

        ext_logits, internal_logits, _, values, _ = self.compute(context, available_flags)
        # values keep all contexts: the last one is the bootstrap value
        # drop the last context for everything else: it is the observation after the last action
        probs_ext = Categorical_With_Mask(logits=ext_logits[:, :-1, :], mask=available_actions)
        internal_entropy = sum(
            Categorical_With_Mask(logits=l[:, :-1, :], mask=m).entropy()
            for l, m in zip(internal_logits, [available_flags, None, None])
        )
        return probs_ext.log_prob(selected_action[:, :, 1]), probs_ext.entropy(), values, internal_entropy


    def unpack_action(self, packed_action):
        # the graph write value is the top recurrent layer's state (the memory's last node_dim entries); see unpack_memory
        int_part, ext_part, position, memory = super().unpack_action(packed_action)
        return int_part, ext_part, position, memory[:, -self.node_dim:]


    def unpack_memory(self, packed_action):
        # (batch, memory_size): [internal head probabilities (feedback_size), every recurrent layer's state, bottom layer first]
        # zeros mean "no previous output" and a fresh state, e.g. at an episode start
        return packed_action[:, 2 + self.position_size:].astype(float)


    def pack_context(self, b_reward=None, b_int=None, b_ext=None, b_position=None, b_memory_reset=None, b_memory=None, b_content=None):
        # b_memory: previous memory (batch, memory_size), see unpack_memory
        # b_memory_reset: (batch,) bool, True where the recurrence restarts from b_memory (e.g. an episode start with zeros)
        batch_size = next(b.shape[0] for b in (b_reward, b_int, b_ext, b_position, b_memory_reset, b_memory, b_content) if b is not None)
        if b_memory is None:
            b_memory = np.zeros((batch_size, self.memory_size), dtype=float)
            b_memory_reset = np.ones((batch_size,), dtype=bool)
        if b_memory_reset is None:
            b_memory_reset = np.zeros((batch_size,), dtype=bool)
        packed_context = super().pack_context(b_reward=b_reward, b_int=b_int, b_ext=b_ext, b_position=b_position, b_content=b_content)
        return np.concatenate([
            packed_context[:, :self.memory_reset_pos],
            np.reshape(b_memory_reset, (batch_size, 1)),
            np.reshape(b_memory, (batch_size, self.memory_size)),
            packed_context[:, self.memory_reset_pos:]
        ], axis=-1).astype(float)


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    C, B, T = 3, 4, 5
    core = Policy_Core(
        int_action_size=6, ext_action_size=8,
        node_dim=8, internal_state_size=1 + C,
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
            np.random.uniform(-1, 1, (B, (1 + C) * 8)),
            np.random.randint(0, 32, (B, 5 + 3)),
            np.random.randint(0, 10, (B, 7 * 7 * 3)),
        ], axis=-1)

    valid = np.random.rand(B, T + 1, 6 + 8) > 0.3
    valid[:, :, 0] = True
    valid[:, :, 6] = True

    # act one step at a time as model_81 does: the context carries the executed action and the memory
    contexts, actions = [], []
    last_action = np.zeros((B, core.packed_action_size))
    for t in range(T + 1):
        context = core.pack_context(
            b_reward=np.zeros(B),
            b_int=last_action[:, 0].astype(int),
            b_ext=last_action[:, 1:2].astype(int),
            b_position=last_action[:, 2:4],
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
    assert (valid[..., :6].gather(-1, actions[..., 0:1].long()).all() and valid[..., 6:].gather(-1, actions[..., 1:2].long()).all()), "sampled actions must be valid"

    # learning slides through time with differentiable feedback, yet reproduces acting exactly (flag masks shape the fed-back probabilities)
    flags = core._split_valid_actions(valid)[0]
    ext_logits, internal_logits, memory_state, values, _ = core.compute(contexts, flags)
    assert torch.allclose(memory_state, actions[:, :, 4:], atol=1e-5), "unrolled memory differs from the acted memory"
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

    # a loss on step 1's ext log-prob reaches the internal heads only through step 0's fed-back probabilities
    def internal_head_grads(contexts):
        core.zero_grad()
        log_probs = core.get_log_probability_with_value(contexts, actions[:, :T], valid[:, :T])[0]
        log_probs[:, 1].sum().backward()
        return [sum(p.grad.abs().sum().item() for p in head.parameters() if p.grad is not None)
                for head in core.recurrent_cell.heads]

    assert all(g > 0 for g in internal_head_grads(contexts)), "internal heads must be trained by later steps' losses"
    restarted = contexts.clone()
    restarted[:, 1, core.memory_reset_pos] = 1.0
    assert all(g == 0 for g in internal_head_grads(restarted)), "a restart must cut the internal-output gradient path"

    print("All tests passed!")
