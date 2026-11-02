import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Beta, Independent
import numpy as np

from interfaces.network import Policy_Value_Network
from implementations.networks.torch.components.base import Categorical_With_Mask
from implementations.networks.torch.components.std_resnet import ResNet
from implementations.networks.torch.components.std_conv import FiLMMiniGridCNN
from implementations.networks.torch.components.wavenet import BasicWavenet
from implementations.networks.torch.components.transformer import InstructionTransformer, get_padding_mask
from implementations.networks.torch.policy.base import Policy_Core as Base_Policy_Core
from implementations.networks.torch.policy.base import Projector as Base_Projector
from utilities.safe_torch_module import Safe_nn_Module


class Policy_Core(Base_Policy_Core, Policy_Value_Network):

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
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="babyai_cont_core", device=device, persistence_path=persistence_path)
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
        # keeps Beta log-probs finite for samples that round to the support boundary
        self.write_eps = 1e-4

        self.position_size = 2
        self.content_size = internal_state_size * node_dim + goal_size + inventory_size + width * height * channel
        self.packed_action_size = 1 + 1 + self.position_size + node_dim
        self.packed_context_size = 1 + 1 + 1 + self.position_size + self.content_size

        self.internal_state_pos = 1 + 1 + 1 + self.position_size
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
            dropout=0.0,
            max_len=goal_size + inventory_size,
            use_pe=True,  # mission semantics depend on word order (attribute binding, "then"/"after")
            pooling="cls",  # mean pooling dilutes order-dependent structure ("A then B" vs "B then A")
        )

        self.internal_state_feature_extraction = ResNet(
            output_dims=hidden_size,
            input_dims=internal_state_size * node_dim,
            hidden_dims=hidden_size,
            layers=[hidden_size for _ in layers]
        )

        # vision conditioned on the instruction and internal state features via FiLM
        self.conv_layers = FiLMMiniGridCNN(
            output_dims=hidden_size, 
            cond_dims=hidden_size + hidden_size,
            input_channels=channel, 
            width=width, height=height,
            vocab_size=256, embedding_dim=8,
            channels=cnn_channels, num_blocks=cnn_blocks
        )

        # internal_state, goal, inv, obs -> hidden
        vec_dim = int_action_size + ext_action_size + hidden_size + hidden_size + hidden_size 
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
        self.head_write_value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2 * node_dim)   # Beta (alpha, beta) per node dimension
        )

        self.head_value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2)
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
        internal_state = context[:, :, self.internal_state_pos: self.goal_pos]  # (batch_size, context_size, internal_state_size * node_dim)
        goal = context[:, :, self.goal_pos: (self.goal_pos + self.goal_size)]  # (batch_size, context_size, goal_size)
        inv = context[:, :, self.inv_pos: (self.inv_pos + self.inventory_size)]  # (batch_size, context_size, inventory_size)
        obs = context[:, :, self.obs_pos: ]  # (batch_size, context_size, obs_size)

        goal_inv = torch.concat([goal, inv], dim=-1)  # (batch_size, context_size, goal_size + inventory_size)
        padding_mask = get_padding_mask(goal_inv, self.pad_token_id)  # (batch_size, context_size, goal_size + inventory_size)
        
        goal_embedded = self.goal_embedding(goal.long())  # (batch_size, context_size, goal_size, embedding_dim)
        inv_embedded = self.inv_embedding(inv.long())  # (batch_size, context_size, inventory_size, embedding_dim)

        goal_inv_embedded = torch.concat([goal_embedded, inv_embedded], dim=2)  # (batch_size, context_size, internal_state_size + goal_size + inventory_size, embedding_dim)
        goal_inv_features = self.goal_feature_extraction(goal_inv_embedded, padding_mask)  # (batch_size, context_size, hidden_size)

        # Extract features from the internal state
        internal_state_features = self.internal_state_feature_extraction(internal_state)  # (batch_size, context_size, hidden_size)

        cond = torch.concat([goal_inv_features, internal_state_features], dim=-1)  # (batch_size, context_size, hidden_size * 2)
    
        obs_features = torch.reshape(obs, (batch_size * context_size, self.height, self.width, self.channel))  # (batch_size * context_size, height, width, channel)
        cond = cond.reshape(batch_size * context_size, 2 * self.hidden_size)
        obs_features = self.conv_layers(obs_features, cond)  # (batch_size * context_size, hidden_size)
        obs_features = obs_features.view(batch_size, context_size, self.hidden_size) # (batch_size, context_size, hidden_size)

        base_input = torch.concat([flag_onehot, action_onehot, goal_inv_features, internal_state_features, obs_features], dim=-1)  # (batch_size, context_size, vec_dim)
        base_output = self.backbone(base_input)  # (batch_size, context_size, hidden_size)

        int_logits = self.head_int(base_output)  # (batch_size, context_size, int_action_size)
        ext_logits = self.head_ext(base_output)  # (batch_size, context_size, ext_action_size)
        edge_1_logits = self.head_edge_1(base_output)  # (batch_size, context_size, C)
        edge_2_logits = self.head_edge_2(base_output)  # (batch_size, context_size, C)
        write_value_params = self.head_write_value(base_output)  # (batch_size, context_size, 2 * node_dim)

        values = self.head_value(base_output)  # (batch_size, context_size, 2)

        return int_logits, ext_logits, edge_1_logits, edge_2_logits, write_value_params, values


    def _split_valid_actions(self, valid_actions):
        if valid_actions is None:
            return None, None
        if isinstance(valid_actions, np.ndarray):
            valid_actions = torch.tensor(valid_actions, dtype=torch.bool)
        valid_actions = valid_actions.to(self.device)
        # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
        return valid_actions[:, :, :self.int_action_size], valid_actions[:, :, self.int_action_size:]


    def _make_distributions(self, logits_int, logits_ext, edge_1_logits, edge_2_logits, write_value_params, available_flags, available_actions):
        alpha_raw, beta_raw = write_value_params.chunk(2, dim=-1)
        # concentrations >= 1 keep each Beta unimodal with bounded density
        alpha = 1.0 + nn.functional.softplus(alpha_raw)
        beta = 1.0 + nn.functional.softplus(beta_raw)
        return (
            Categorical_With_Mask(logits=logits_int, mask=available_flags),
            Categorical_With_Mask(logits=logits_ext, mask=available_actions),
            Categorical(logits=edge_1_logits),
            Categorical(logits=edge_2_logits),
            Independent(Beta(alpha, beta), 1),  # joint over node_dim
        )


    def _log_prob_entropy(self, dists, selected_action):
        probs_int, probs_ext, probs_edge_1, probs_edge_2, probs_write_value = dists
        action_write_value = selected_action[:, :, 4:4 + self.node_dim].clamp(self.write_eps, 1.0 - self.write_eps)

        log_probs = torch.stack([
            probs_int.log_prob(selected_action[:, :, 0]),
            probs_ext.log_prob(selected_action[:, :, 1]),
            probs_edge_1.log_prob(selected_action[:, :, 2]),
            probs_edge_2.log_prob(selected_action[:, :, 3]),
            probs_write_value.log_prob(action_write_value),
        ], dim=-1)
        entropy = torch.stack([
            probs_int.entropy(),
            probs_ext.entropy(),
            probs_edge_1.entropy(),
            probs_edge_2.entropy(),
            # mean, not sum: keeps the regularizer scale independent of node_dim (log-prob stays joint)
            probs_write_value.base_dist.entropy().mean(dim=-1),
        ], dim=-1)
        return log_probs, entropy
    

    def get_action(self, context, valid_actions=None):

        if isinstance(context, np.ndarray):
            # float: node values are continuous and would be truncated by a long cast
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        available_flags, available_actions = self._split_valid_actions(valid_actions)

        with torch.no_grad():
            outputs = self.compute(context)
            probs_int, probs_ext, probs_edge_1, probs_edge_2, probs_write_value = self._make_distributions(
                *outputs[:5], available_flags, available_actions
            )

            action_int = probs_int.sample()
            action_ext = probs_ext.sample()
            edge_1 = probs_edge_1.sample()
            edge_2 = probs_edge_2.sample()
            write_value = probs_write_value.sample().clamp(self.write_eps, 1.0 - self.write_eps)  # (batch, context_size, node_dim)

        action = np.concatenate([
            action_int.unsqueeze(-1).cpu().numpy(),
            action_ext.unsqueeze(-1).cpu().numpy(),
            edge_1.unsqueeze(-1).cpu().numpy(),
            edge_2.unsqueeze(-1).cpu().numpy(),
            write_value.cpu().numpy()
        ], axis=-1)

        return action.astype(float)
    

    def get_log_probability(self, context, selected_action, valid_actions=None):
        # context has shape (batch, context_size, self.packed_context_size)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        available_flags, available_actions = self._split_valid_actions(valid_actions)

        outputs = self.compute(context)
        dists = self._make_distributions(*outputs[:5], available_flags, available_actions)
        return self._log_prob_entropy(dists, selected_action)


    def get_log_probability_with_aux_loss(self, context, selected_action, valid_actions=None):
        # now context has shape (batch, context_size + 1, self.packed_context_size)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        context = context[:, :-1, :]  # remove last context for computing logprob

        available_flags, available_actions = self._split_valid_actions(valid_actions)

        outputs = self.compute(context)
        dists = self._make_distributions(*outputs[:5], available_flags, available_actions)
        log_probs, entropy = self._log_prob_entropy(dists, selected_action)
        return log_probs, entropy, None


    def get_log_probability_with_value(self, context, selected_action, valid_actions=None):
        # now context has shape (batch, context_size + 1, self.packed_context_size)
        # Return log prob will only have context_size, but value will have context_size + 1

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        available_flags, available_actions = self._split_valid_actions(valid_actions)

        outputs = self.compute(context)
        values = outputs[5]  # keep all contexts: the last one is the bootstrap value
        # drop the last context for logprob: it is the observation after the last action
        dists = self._make_distributions(*(o[:, :-1, :] for o in outputs[:5]), available_flags, available_actions)
        log_probs, entropy = self._log_prob_entropy(dists, selected_action)
        return log_probs, entropy, values


class Projector(Base_Projector):
    
    def get_log_probability_with_value(self, context, selected_action, valid_actions=None):
        all_logprobs, all_entropy, values = self.master_core.get_log_probability_with_value(context, selected_action, valid_actions)
        log_probs = all_logprobs[:, :, self.selected_indices].sum(dim=-1)  # sum over selected logprob components
        entropy = all_entropy[:, :, self.selected_indices].sum(dim=-1)
        return log_probs, entropy, values
    