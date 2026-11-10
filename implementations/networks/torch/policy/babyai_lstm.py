import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

from interfaces.network import Actor_Network, Policy_Value_Network
from implementations.networks.torch.components.base import Categorical_With_Mask
from implementations.networks.torch.components.std_resnet import ResNet
from implementations.networks.torch.components.std_conv import FiLMMiniGridCNN
from implementations.networks.torch.components.std_lstm import LSTM
from implementations.networks.torch.components.transformer import InstructionTransformer, get_padding_mask
from implementations.networks.torch.policy.base import Policy_Core as Base_Policy_Core
from implementations.networks.torch.policy.babyai import Projector  # noqa: F401 -- per-component masking; use with model_76.COMPONENT_FLAGS
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
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="babyai_lstm_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.int_action_size = int_action_size  # num classes for flag
        self.ext_action_size = ext_action_size
        self.internal_state_size = internal_state_size
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
        # the write value is the LSTM cell state, not sampled
        self.packed_action_size = 1 + 1 + self.position_size + node_dim
        # previous LSTM cell state (+ a flag marking where it restarts) rides in the context:
        # acting reads it per step, learning slides the LSTM through time and only restarts from it at flagged steps
        self.packed_context_size = 1 + 1 + 1 + self.position_size + 1 + node_dim + self.content_size

        self.memory_reset_pos = 1 + 1 + 1 + self.position_size
        self.memory_pos = self.memory_reset_pos + 1
        self.internal_state_pos = self.memory_pos + node_dim
        self.goal_pos = self.internal_state_pos + self.internal_state_size * node_dim
        self.inv_pos = self.goal_pos + goal_size
        self.obs_pos = self.inv_pos + inventory_size

        self.pad_token_id = pad_token_id

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
            cond_dim=hidden_size,  # internal state token lets memory select which mission clause to attend
        )

        self.internal_state_feature_extraction = ResNet(
            output_dims=hidden_size,
            input_dims=internal_state_size * node_dim,
            hidden_dims=hidden_size,
            layers=[hidden_size for _ in layers]
        )

        # vision conditioned via FiLM on the internal-state-aware instruction features
        self.conv_layers = FiLMMiniGridCNN(
            output_dims=hidden_size, 
            cond_dims=hidden_size,
            input_channels=channel, 
            width=width, height=height,
            vocab_size=256, embedding_dim=8,
            channels=cnn_channels, num_blocks=cnn_blocks
        )

        # internal_state, goal, inv, obs (+ previous cell state) -> hidden, cell state
        vec_dim = int_action_size + ext_action_size + hidden_size + hidden_size + hidden_size 
        self.backbone = LSTM(
            output_dims=hidden_size, 
            input_dims=vec_dim, 
            hidden_dims=node_dim, 
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
        self.head_edge_1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, internal_state_size - 1)
        )
        self.head_edge_2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, internal_state_size - 1)
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
        self.head_edge_1.apply(init_actor_weights)
        self.head_edge_2.apply(init_actor_weights)

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
        memory_reset = context[:, :, self.memory_reset_pos] > 0.5  # (batch_size, context_size)
        prev_memory_state = context[:, :, self.memory_pos: self.internal_state_pos]  # (batch_size, context_size, node_dim)
        internal_state = context[:, :, self.internal_state_pos: self.goal_pos]  # (batch_size, context_size, internal_state_size * node_dim)
        goal = context[:, :, self.goal_pos: (self.goal_pos + self.goal_size)]  # (batch_size, context_size, goal_size)
        inv = context[:, :, self.inv_pos: (self.inv_pos + self.inventory_size)]  # (batch_size, context_size, inventory_size)
        obs = context[:, :, self.obs_pos: ]  # (batch_size, context_size, obs_size)

        internal_state_features = self.internal_state_feature_extraction(internal_state)  # (batch_size, context_size, hidden_size)

        goal_inv = torch.concat([goal, inv], dim=-1)  # (batch_size, context_size, goal_size + inventory_size)
        padding_mask = get_padding_mask(goal_inv, self.pad_token_id)  # (batch_size, context_size, goal_size + inventory_size)
        
        goal_embedded = self.goal_embedding(goal.long())  # (batch_size, context_size, goal_size, embedding_dim)
        inv_embedded = self.inv_embedding(inv.long())  # (batch_size, context_size, inventory_size, embedding_dim)

        goal_inv_embedded = torch.concat([goal_embedded, inv_embedded], dim=2)  # (batch_size, context_size, goal_size + inventory_size, embedding_dim)
        goal_inv_features = self.goal_feature_extraction(goal_inv_embedded, padding_mask, cond=internal_state_features)  # (batch_size, context_size, hidden_size)

        obs_features = torch.reshape(obs, (batch_size * context_size, self.height, self.width, self.channel))  # (batch_size * context_size, height, width, channel)
        cond = goal_inv_features.reshape(batch_size * context_size, self.hidden_size)
        obs_features = self.conv_layers(obs_features, cond)  # (batch_size * context_size, hidden_size)
        obs_features = obs_features.view(batch_size, context_size, self.hidden_size) # (batch_size, context_size, hidden_size)

        base_input = torch.concat([flag_onehot, action_onehot, goal_inv_features, internal_state_features, obs_features], dim=-1)  # (batch_size, context_size, vec_dim)
        base_output, memory_state = self.backbone(base_input, prev_memory_state, memory_reset)  # (batch_size, context_size, hidden_size), (batch_size, context_size, node_dim)

        int_logits = self.head_int(base_output)  # (batch_size, context_size, int_action_size)
        ext_logits = self.head_ext(base_output)  # (batch_size, context_size, ext_action_size)
        edge_1_logits = self.head_edge_1(base_output)  # (batch_size, context_size, C)
        edge_2_logits = self.head_edge_2(base_output)  # (batch_size, context_size, C)

        values = self.head_value(base_output)  # (batch_size, context_size, value_size)
        if self.value_size == 1:
            values = values.squeeze(-1)  # (batch_size, context_size)

        return int_logits, ext_logits, edge_1_logits, edge_2_logits, memory_state, values


    def _split_valid_actions(self, valid_actions):
        if valid_actions is None:
            return None, None
        if isinstance(valid_actions, np.ndarray):
            valid_actions = torch.tensor(valid_actions, dtype=torch.bool)
        valid_actions = valid_actions.to(self.device)
        # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
        return valid_actions[:, :, :self.int_action_size], valid_actions[:, :, self.int_action_size:]


    def _make_distributions(self, logits_int, logits_ext, edge_1_logits, edge_2_logits, available_flags, available_actions):
        return (
            Categorical_With_Mask(logits=logits_int, mask=available_flags),
            Categorical_With_Mask(logits=logits_ext, mask=available_actions),
            Categorical(logits=edge_1_logits),
            Categorical(logits=edge_2_logits),
        )


    def _log_prob_entropy(self, dists, selected_action):
        probs_int, probs_ext, probs_edge_1, probs_edge_2 = dists

        log_probs = torch.stack([
            probs_int.log_prob(selected_action[:, :, 0]),
            probs_ext.log_prob(selected_action[:, :, 1]),
            probs_edge_1.log_prob(selected_action[:, :, 2]),
            probs_edge_2.log_prob(selected_action[:, :, 3])
        ], dim=-1)
        entropy = torch.stack([
            probs_int.entropy(),
            probs_ext.entropy(),
            probs_edge_1.entropy(),
            probs_edge_2.entropy()
        ], dim=-1)
        return log_probs, entropy
    

    def get_action(self, context, valid_actions=None):

        if isinstance(context, np.ndarray):
            # float: node values are continuous and would be truncated by a long cast
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        available_flags, available_actions = self._split_valid_actions(valid_actions)

        with torch.no_grad():
            outputs = self.compute(context)
            probs_int, probs_ext, probs_edge_1, probs_edge_2 = self._make_distributions(
                *outputs[:4], available_flags, available_actions
            )
            write_value = outputs[4]  # (batch, context_size, node_dim)

            action_int = probs_int.sample()
            action_ext = probs_ext.sample()
            edge_1 = probs_edge_1.sample()
            edge_2 = probs_edge_2.sample()

        action = np.concatenate([
            action_int.unsqueeze(-1).cpu().numpy(),
            action_ext.unsqueeze(-1).cpu().numpy(),
            edge_1.unsqueeze(-1).cpu().numpy(),
            edge_2.unsqueeze(-1).cpu().numpy(),
            write_value.cpu().numpy()
        ], axis=-1)

        return action.astype(float)
    

    def get_log_prob_with_value_aux(self, context, selected_action, valid_actions=None):
        log_probs, entropy, values = self.get_log_probability_with_value(context, selected_action, valid_actions)
        return log_probs, entropy, values, None  # no auxiliary loss


    def get_log_probability_with_value(self, context, selected_action, valid_actions=None):
        # now context has shape (batch, context_size + 1, self.packed_context_size)
        # Return log prob will only have context_size, but value will have context_size + 1

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        available_flags, available_actions = self._split_valid_actions(valid_actions)

        outputs = self.compute(context)
        values = outputs[5]  # keep all contexts: the last one is the bootstrap value
        # drop the last context for logprob: it is the observation after the last action
        dists = self._make_distributions(*(o[:, :-1, :] for o in outputs[:4]), available_flags, available_actions)
        log_probs, entropy = self._log_prob_entropy(dists, selected_action)
        return log_probs, entropy, values


    def pack_context(self, b_reward=None, b_int=None, b_ext=None, b_position=None, b_memory_reset=None, b_memory=None, b_content=None):
        # b_memory: previous LSTM cell state (batch, node_dim)
        # b_memory_reset: (batch,) bool, True where the recurrence restarts from b_memory (e.g. an episode start with zeros)
        batch_size = next(b.shape[0] for b in (b_reward, b_int, b_ext, b_position, b_memory_reset, b_memory, b_content) if b is not None)
        if b_memory is None:
            b_memory = np.zeros((batch_size, self.node_dim), dtype=float)
            b_memory_reset = np.ones((batch_size,), dtype=bool)
        if b_memory_reset is None:
            b_memory_reset = np.zeros((batch_size,), dtype=bool)
        packed_context = super().pack_context(b_reward=b_reward, b_int=b_int, b_ext=b_ext, b_position=b_position, b_content=b_content)
        return np.concatenate([
            packed_context[:, :self.memory_reset_pos],
            np.reshape(b_memory_reset, (batch_size, 1)),
            np.reshape(b_memory, (batch_size, self.node_dim)),
            packed_context[:, self.memory_reset_pos:]
        ], axis=-1).astype(float)
