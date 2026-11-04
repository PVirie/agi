import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from implementations.networks.torch.components.std_resnet import ResNet
from implementations.networks.torch.components.std_conv import FiLMMiniGridCNN
from implementations.networks.torch.components.wavenet import BasicWavenet
from implementations.networks.torch.components.transformer import InstructionTransformer, get_padding_mask
from implementations.networks.torch.components.base import Categorical_With_Mask
from implementations.networks.torch.policy.minigrid import Policy_Core as Minigrid_Policy_Core
from implementations.networks.torch.policy.babyai import Projector  # noqa: F401 -- same component masking; use with model_77.COMPONENT_FLAGS
from utilities.safe_torch_module import Safe_nn_Module


class Policy_Core(Minigrid_Policy_Core):

    def __init__(self, 
                 int_action_size, ext_action_size,
                 write_action_size,
                 internal_state_size,
                 goal_size,
                 inventory_size,
                 width, height, channel,
                 dict_size, embedding_dim, pad_token_id,
                 hidden_size, layers, 
                 cnn_channels=128, cnn_blocks=2,
                 value_size=1,
                 location_size=64,
                 bridge_flag=2,
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="babyai_77_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.int_action_size = int_action_size  # num classes for flag
        self.ext_action_size = ext_action_size
        self.write_action_size = write_action_size

        self.internal_state_size = internal_state_size
        self.goal_size = goal_size
        self.inventory_size = inventory_size
        self.width = width
        self.height = height
        self.channel = channel
        self.dict_size = dict_size
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        # 1 for single-stream PPO (values squeezed to (batch, context)); 2 for ppo_graph_dual [extrinsic, intrinsic]
        self.value_size = value_size

        # bridge: each step emits a query/key "location vector"; q_t . k_j picks a previous step j to bridge to
        self.location_size = location_size
        # int-action flag that performs the bridge (model_77 INT_FLAG_OPS); the bridge target only counts on these steps
        self.bridge_flag = bridge_flag
        self._bridge_key_cache = None  # rollout-only cache of past keys, see get_action

        self.position_size = 2
        self.content_size = internal_state_size + goal_size + inventory_size + width * height * channel
        self.packed_action_size = 1 + 1 + self.position_size + 1
        self.packed_context_size = 1 + 1 + 1 + self.position_size + self.content_size

        self.internal_state_pos = 1 + 1 + 1 + self.position_size
        self.goal_pos = self.internal_state_pos + self.internal_state_size
        self.inv_pos = self.goal_pos + goal_size
        self.obs_pos = self.inv_pos + inventory_size

        self.pad_token_id = pad_token_id

        self.internal_state_embedding = nn.Embedding(dict_size, embedding_dim, padding_idx=pad_token_id)  # for internal state tokens
        self.goal_embedding = nn.Embedding(dict_size, embedding_dim, padding_idx=pad_token_id)  # for goal tokens
        self.inv_embedding = nn.Embedding(dict_size, embedding_dim, padding_idx=pad_token_id)  # for inventory tokens

        # self.goal_wavenet = BasicWavenet(d_model=embedding_dim, num_layers=2)

        self.internal_state_feature_extraction = InstructionTransformer(
            input_dim=embedding_dim,
            d_model=hidden_size,
            nhead=8,
            num_layers=2,
            dropout=0.0,
            max_len=internal_state_size + goal_size + inventory_size,
            use_pe=True,  # mission semantics depend on word order (attribute binding, "then"/"after")
            pooling="cls",  # mean pooling dilutes order-dependent structure ("A then B" vs "B then A")
        )

        # vision conditioned on the instruction/internal state features via FiLM
        self.conv_layers = FiLMMiniGridCNN(
            output_dims=hidden_size, 
            cond_dims=hidden_size,
            input_channels=channel, 
            width=width, height=height,
            vocab_size=256, embedding_dim=8,
            channels=cnn_channels, num_blocks=cnn_blocks
        )

        # internal_state, goal, inv, obs -> hidden
        vec_dim = int_action_size + ext_action_size + hidden_size + hidden_size 
        self.backbone = ResNet(
            output_dims=hidden_size, 
            input_dims=vec_dim, 
            hidden_dims=hidden_size, 
            layers=[hidden_size for _ in layers]
        )

        self.head_int = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, int_action_size)   # int_action_size classes
        )
        self.head_ext = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, ext_action_size)   # ext_action_size classes
        )
        self.head_write_value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.write_action_size)
        )
        self.head_bridge_k = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, location_size)
        )
        self.head_bridge_q = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, location_size)
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
        self.internal_state_embedding.reset_parameters()
        self.goal_embedding.reset_parameters()
        self.inv_embedding.reset_parameters()

        # self.goal_wavenet.reset_parameters()

        self.internal_state_feature_extraction.reset_parameters()

        self.conv_layers.reset_parameters()

        self.backbone.reset_parameters()

        def init_actor_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.head_int.apply(init_actor_weights)
        self.head_ext.apply(init_actor_weights)
        self.head_bridge_k.apply(init_actor_weights)
        self.head_bridge_q.apply(init_actor_weights)
        self.head_write_value.apply(init_actor_weights)

        def init_value_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        self.head_value.apply(init_value_weights)


    def compute(self, context):
        # context has shape (batch, context_size, self.packed_context_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        flag_onehot = torch.nn.functional.one_hot(context[:, :, 1].long(), num_classes=self.int_action_size).float()
        action_onehot = torch.nn.functional.one_hot(context[:, :, 2].long(), num_classes=self.ext_action_size).float()
        internal_state = context[:, :, self.internal_state_pos: (self.internal_state_pos + self.internal_state_size)]  # (batch_size, context_size, internal_state_size)
        goal = context[:, :, self.goal_pos: (self.goal_pos + self.goal_size)]  # (batch_size, context_size, goal_size)
        inv = context[:, :, self.inv_pos: (self.inv_pos + self.inventory_size)]  # (batch_size, context_size, inventory_size)
        obs = context[:, :, self.obs_pos: ]  # (batch_size, context_size, obs_size)

        internal_state_goal_inv = torch.concat([internal_state, goal, inv], dim=-1)  # (batch_size, context_size, internal_state_size + goal_size + inventory_size)
        padding_mask = get_padding_mask(internal_state_goal_inv, self.pad_token_id)  # (batch_size, context_size, internal_state_size + goal_size + inventory_size)
        
        internal_state_embedded = self.internal_state_embedding(internal_state.long())  # (batch_size, context_size, internal_state_size, embedding_dim)
        goal_embedded = self.goal_embedding(goal.long())  # (batch_size, context_size, goal_size, embedding_dim)
        inv_embedded = self.inv_embedding(inv.long())  # (batch_size, context_size, inventory_size, embedding_dim)

        # # run wavenet on goal to get a better representation of the goal sequence
        # goal_embedded = goal_embedded.view(batch_size * context_size, self.goal_size, self.embedding_dim)  # (batch_size * context_size, goal_size, embedding_dim)
        # goal_embedded = self.goal_wavenet(goal_embedded)  # (batch_size * context_size, goal_size, embedding_dim)
        # goal_embedded = goal_embedded.view(batch_size, context_size, self.goal_size, self.embedding_dim)  # (batch_size, context_size, goal_size, embedding_dim)

        internal_state_goal_embedded = torch.concat([internal_state_embedded, goal_embedded, inv_embedded], dim=2)  # (batch_size, context_size, internal_state_size + goal_size + inventory_size, embedding_dim)
        internal_state_features = self.internal_state_feature_extraction(internal_state_goal_embedded, padding_mask)  # (batch_size, context_size, hidden_size)

        obs_features = torch.reshape(obs, (batch_size * context_size, self.height, self.width, self.channel))  # (batch_size * context_size, height, width, channel)
        cond = internal_state_features.reshape(batch_size * context_size, self.hidden_size)
        obs_features = self.conv_layers(obs_features, cond)  # (batch_size * context_size, hidden_size)
        obs_features = obs_features.view(batch_size, context_size, self.hidden_size) # (batch_size, context_size, hidden_size)

        base_input = torch.concat([flag_onehot, action_onehot, internal_state_features, obs_features], dim=-1)  # (batch_size, context_size, vec_dim)
        base_output = self.backbone(base_input)  # (batch_size, context_size, hidden_size)

        int_logits = self.head_int(base_output)  # (batch_size, context_size, int_action_size)
        ext_logits = self.head_ext(base_output)  # (batch_size, context_size, ext_action_size)
        bridge_q = self.head_bridge_q(base_output)  # (batch_size, context_size, location_size)
        bridge_k = self.head_bridge_k(base_output)  # (batch_size, context_size, location_size)
        write_value_logits = self.head_write_value(base_output)  # (batch_size, context_size, write_action_size)

        values = self.head_value(base_output)  # (batch_size, context_size, value_size)
        if self.value_size == 1:
            values = values.squeeze(-1)  # (batch_size, context_size)

        return int_logits, ext_logits, bridge_q, bridge_k, write_value_logits, values


    def _bridge_logits(self, q, k, offset=0):
        # q: (B, Tq, d) queries for window steps offset .. offset + Tq - 1; k: (B, Tk, d) keys for window steps 0 .. Tk - 1
        # returns logits (B, Tq, Tk) over which window step to bridge to, and the causal mask (step i sees keys j <= i)
        logits = torch.einsum("bqd,bkd->bqk", q, k) / math.sqrt(self.location_size)
        q_pos = (torch.arange(q.size(1), device=q.device)[:, None] + offset).clamp(min=0)
        k_pos = torch.arange(k.size(1), device=q.device)[None, :]
        mask = (k_pos <= q_pos).unsqueeze(0).expand(q.size(0), -1, -1)
        return logits, mask


    def reset_bridge_cache(self):
        self._bridge_key_cache = None


    def train(self, mode=True):
        # entering train mode means learn() is about to change the parameters and the window restarts afterwards
        # (graph memory reset_timestamp), so cached keys are stale either way
        if mode:
            self.reset_bridge_cache()
        return super().train(mode)


    def _split_valid_actions(self, valid_actions):
        if valid_actions is None:
            return None, None
        if isinstance(valid_actions, np.ndarray):
            valid_actions = torch.tensor(valid_actions, dtype=torch.bool)
        valid_actions = valid_actions.to(self.device)
        # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
        return valid_actions[:, :, :self.int_action_size], valid_actions[:, :, self.int_action_size:]


    def _distributions(self, int_logits, ext_logits, bridge_q, bridge_k, write_value_logits, valid_actions):
        available_flags, available_actions = self._split_valid_actions(valid_actions)
        bridge_logits, bridge_mask = self._bridge_logits(bridge_q, bridge_k)
        return (
            Categorical_With_Mask(logits=int_logits, mask=available_flags),
            Categorical_With_Mask(logits=ext_logits, mask=available_actions),
            Categorical_With_Mask(logits=bridge_logits, mask=bridge_mask),
            Categorical(logits=write_value_logits),
        )


    def _log_prob_and_entropy(self, dists, selected_action):
        probs_int, probs_ext, probs_bridge, probs_write_value = dists

        action_int = selected_action[:, :, 0]
        action_ext = selected_action[:, :, 1]
        action_bridge_step = selected_action[:, :, 1 + 1 + 0]  # position[0] holds the sampled window step
        action_write_value = selected_action[:, :, 1 + 1 + 2]

        # the bridge target only affects the outcome when the flag is BRIDGE
        is_bridge = (action_int.long() == self.bridge_flag).float()

        log_prob_int = probs_int.log_prob(action_int)
        log_prob_ext = probs_ext.log_prob(action_ext)
        log_prob_bridge = probs_bridge.log_prob(action_bridge_step) * is_bridge
        log_prob_write_value = probs_write_value.log_prob(action_write_value)

        entropy_int = probs_int.entropy()
        entropy_ext = probs_ext.entropy()
        entropy_bridge = probs_bridge.entropy() * is_bridge
        entropy_write_value = probs_write_value.entropy()

        # keep the 5-slot layout [int, ext, position 0, position 1, write]; position 1 is unused
        zeros = torch.zeros_like(log_prob_int)
        return torch.stack([
            log_prob_int, log_prob_ext, log_prob_bridge, zeros, log_prob_write_value
        ], dim=-1), torch.stack([
            entropy_int, entropy_ext, entropy_bridge, zeros, entropy_write_value
        ], dim=-1)


    @torch.no_grad()
    def get_action(self, context, valid_actions=None):
        """
        context: (batch, T, packed_context_size); context[:, -1] is the new step.
        Each call is one new window step: its key is appended to a rollout cache, and the bridge candidates are
        every cached step of the current window (the window restarts when the core enters train mode, i.e. at learn()).
        The cached keys equal those recomputed at training time since parameters only change at learn().

        returns action (batch, T, packed_action_size) = [int, ext, bridge window step, 0, write_value];
        the graph memory resolves the bridge window step to a node through its own head track.
        """
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.long).to(self.device)

        batch_size = context.size(0)
        context_size = context.size(1)

        int_logits, ext_logits, bridge_q, bridge_k, write_value_logits, _ = self.compute(context)

        cache = self._bridge_key_cache
        if cache is not None and cache.size(0) != batch_size:
            raise RuntimeError(f"bridge key cache batch {cache.size(0)} does not match context batch {batch_size}; call reset_bridge_cache()")
        new_key = bridge_k[:, -1:]
        bridge_k = new_key if cache is None else torch.cat([cache, new_key], dim=1)  # (batch, window_size, location_size)
        self._bridge_key_cache = bridge_k

        available_flags, available_actions = self._split_valid_actions(valid_actions)
        bridge_logits, bridge_mask = self._bridge_logits(bridge_q, bridge_k, offset=bridge_k.size(1) - context_size)

        action_int = Categorical_With_Mask(logits=int_logits, mask=available_flags).sample()
        action_ext = Categorical_With_Mask(logits=ext_logits, mask=available_actions).sample()
        bridge_step = Categorical_With_Mask(logits=bridge_logits, mask=bridge_mask).sample()  # (batch, T) window step
        write_value = Categorical(logits=write_value_logits).sample()

        action = np.concatenate([
            action_int.unsqueeze(-1).cpu().numpy(),
            action_ext.unsqueeze(-1).cpu().numpy(),
            bridge_step.unsqueeze(-1).cpu().numpy(),
            np.zeros((batch_size, context_size, 1)),
            write_value.unsqueeze(-1).cpu().numpy()
        ], axis=-1)

        return action.astype(float)


    def get_log_probability(self, context, selected_action, valid_actions=None):
        # context has shape (batch, context_size, self.packed_context_size); window step indices are positions in context
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.long).to(self.device)

        int_logits, ext_logits, bridge_q, bridge_k, write_value_logits, _ = self.compute(context)
        dists = self._distributions(int_logits, ext_logits, bridge_q, bridge_k, write_value_logits, valid_actions)
        return self._log_prob_and_entropy(dists, selected_action)


    def get_log_probability_with_aux_loss(self, context, selected_action, valid_actions=None):
        # now context has shape (batch, context_size + 1, self.packed_context_size)
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.long).to(self.device)

        log_probs, entropy = self.get_log_probability(context[:, :-1, :], selected_action, valid_actions)
        return log_probs, entropy, None


    def get_log_probability_with_value(self, context, selected_action, valid_actions=None):
        # now context has shape (batch, context_size + 1, self.packed_context_size)
        # Return log prob will only have context_size, but value will have context_size + 1
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.long).to(self.device)

        int_logits, ext_logits, bridge_q, bridge_k, write_value_logits, values = self.compute(context)

        # drop the last step's outputs: it is the observation after the last action
        # (step i only attends to keys j <= i, so dropping the last key changes nothing for the remaining rows)
        dists = self._distributions(
            int_logits[:, :-1], ext_logits[:, :-1],
            bridge_q[:, :-1], bridge_k[:, :-1],
            write_value_logits[:, :-1],
            valid_actions,
        )
        log_probs, entropy = self._log_prob_and_entropy(dists, selected_action)
        # but maintain values for all contexts, since we need the bootstrap value for the last context
        return log_probs, entropy, values
