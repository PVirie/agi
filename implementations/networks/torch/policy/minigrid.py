import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np
import logging

from interfaces.network import Policy_Value_Network
from implementations.networks.torch.components.base import init_weights
from implementations.networks.torch.components.base import Categorical_With_Mask
from implementations.networks.torch.components.std_resnet import ResNet
from implementations.networks.torch.components.std_conv import ImpalaCNN
from implementations.networks.torch.components.transformer import InstructionTransformer, get_padding_mask
from implementations.networks.torch.policy.base import Policy_Core as Base_Policy_Core
from implementations.networks.torch.policy.base import Projector as Base_Projector
from utilities.safe_torch_module import Safe_nn_Module


class Policy_Core(Base_Policy_Core, Policy_Value_Network):

    def __init__(self, 
                 int_action_size, ext_action_size,
                 write_action_size,
                 internal_state_size,
                 goal_size,
                 inventory_size,
                 width, height, channel,
                 dict_size, embedding_dim, pad_token_id,
                 hidden_size, layers, 
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="minigrid_core", device=device, persistence_path=persistence_path)
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

        self.internal_state_feature_extraction = InstructionTransformer(
            input_dim=embedding_dim,
            d_model=hidden_size,
            nhead=8, 
            num_layers=4, 
            max_len=internal_state_size + goal_size + inventory_size,
        )

        self.image_embedding = nn.Embedding(256, 4)  # for image pixels, shared across channels
        self.feature_channel = self.channel * 4
        self.conv_layers = ImpalaCNN(
            output_dims=hidden_size, 
            input_channels=self.feature_channel, 
            width=width, height=height,
            depths=layers
        )

        # internal_state, goal, inv, obs -> hidden
        self.backbone = ResNet(
            output_dims=hidden_size, 
            input_dims=hidden_size + hidden_size, 
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
            nn.Linear(hidden_size, self.write_action_size)
        )

        self.head_value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

        self.reset_parameters()
        self.load(override_persistence_path=first_load_path)
        self.eval()


    def reset_parameters(self):
        # Reset parameters of all layers
        self.internal_state_embedding.reset_parameters()
        self.goal_embedding.reset_parameters()
        self.inv_embedding.reset_parameters()

        self.internal_state_feature_extraction.reset_parameters()

        self.image_embedding.reset_parameters()
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
        internal_state = context[:, :, self.internal_state_pos: (self.internal_state_pos + self.internal_state_size)]  # (batch_size, context_size, internal_state_size)
        goal = context[:, :, self.goal_pos: (self.goal_pos + self.goal_size)]  # (batch_size, context_size, goal_size)
        inv = context[:, :, self.inv_pos: (self.inv_pos + self.inventory_size)]  # (batch_size, context_size, inventory_size)
        obs = context[:, :, self.obs_pos: ]  # (batch_size, context_size, obs_size)

        internal_state_goal_inv = torch.concat([internal_state, goal, inv], dim=-1)  # (batch_size, context_size, internal_state_size + goal_size + inventory_size)
        padding_mask = get_padding_mask(internal_state_goal_inv, self.pad_token_id)  # (batch_size, context_size, internal_state_size + goal_size + inventory_size)
        
        internal_state_embedded = self.internal_state_embedding(internal_state.long())  # (batch_size, context_size, internal_state_size, embedding_dim)
        goal_embedded = self.goal_embedding(goal.long())  # (batch_size, context_size, goal_size, embedding_dim)
        inv_embedded = self.inv_embedding(inv.long())  # (batch_size, context_size, inventory_size, embedding_dim)

        internal_state_goal_embedded = torch.concat([internal_state_embedded, goal_embedded, inv_embedded], dim=2)  # (batch_size, context_size, internal_state_size + goal_size + inventory_size, embedding_dim)
        internal_state_features = self.internal_state_feature_extraction(internal_state_goal_embedded, padding_mask)  # (batch_size, context_size, hidden_size)

        obs_embedded = self.image_embedding(obs.long())  # (batch_size, context_size, obs_size, embedding_dim)
        obs_features = torch.reshape(obs_embedded, (batch_size * context_size, self.height, self.width, self.feature_channel))  # (batch_size * context_size, height, width, channel * embedding_dim)
        obs_features = obs_features.permute(0, 3, 1, 2)  # (batch_size * context_size, channel * embedding_dim, height, width)
        obs_features = self.conv_layers(obs_features)  # (batch_size * context_size, hidden_size)
        obs_features = obs_features.view(batch_size, context_size, self.hidden_size) # (batch_size, context_size, hidden_size)

        base_input = torch.concat([internal_state_features, obs_features], dim=-1)  # (batch_size, context_size, vec_dim)
        base_output = self.backbone(base_input)  # (batch_size, context_size, hidden_size)

        int_logits = self.head_int(base_output)  # (batch_size, context_size, int_action_size)
        ext_logits = self.head_ext(base_output)  # (batch_size, context_size, ext_action_size)
        edge_1_logits = self.head_edge_1(base_output)  # (batch_size, context_size, C)
        edge_2_logits = self.head_edge_2(base_output)  # (batch_size, context_size, C)
        write_value_logits = self.head_write_value(base_output)  # (batch_size, context_size, write_action_size)

        values = self.head_value(base_output)  # (batch_size, context_size, 1)
        values = values.squeeze(-1)  # (batch_size, context_size)

        return int_logits, ext_logits, edge_1_logits, edge_2_logits, write_value_logits, values
    

    def get_action(self, context, valid_actions=None):

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.long).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
            available_flags = valid_actions[:, :, :self.int_action_size].to(self.device)
            available_actions = valid_actions[:, :, self.int_action_size:].to(self.device)

        logits_int, logits_ext, edge_1_logits, edge_2_logits, write_value_logits, _ = self.compute(context)

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)
        probs_edge_1 = Categorical(logits=edge_1_logits)
        probs_edge_2 = Categorical(logits=edge_2_logits)
        probs_write_value = Categorical(logits=write_value_logits)

        batch_size = context.size(0)
        context_size = context.size(1)

        action_int = probs_int.sample()
        action_ext = probs_ext.sample()
        edge_1 = probs_edge_1.sample()
        edge_2 = probs_edge_2.sample()
        write_value = probs_write_value.sample()

        action = np.concatenate([
            action_int.unsqueeze(-1).cpu().numpy(),
            action_ext.unsqueeze(-1).cpu().numpy(),
            edge_1.unsqueeze(-1).cpu().numpy(),
            edge_2.unsqueeze(-1).cpu().numpy(),
            write_value.unsqueeze(-1).cpu().numpy()
        ], axis=-1)

        return action.astype(float)
    

    def get_log_probability(self, context, selected_action, valid_actions=None):
        # context has shape (batch, context_size, self.packed_context_size)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.long).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
            available_flags = valid_actions[:, :, :self.int_action_size].to(self.device)
            available_actions = valid_actions[:, :, self.int_action_size:].to(self.device)

        logits_int, logits_ext, edge_1_logits, edge_2_logits, write_value_logits, _ = self.compute(context)

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)
        probs_edge_1 = Categorical(logits=edge_1_logits)
        probs_edge_2 = Categorical(logits=edge_2_logits)
        probs_write_value = Categorical(logits=write_value_logits)

        action_int = selected_action[:, :, 0]
        action_ext = selected_action[:, :, 1]
        action_edge_1 = selected_action[:, :, 1 + 1 + 0]
        action_edge_2 = selected_action[:, :, 1 + 1 + 1]
        action_write_value = selected_action[:, :, 1 + 1 + 2]

        log_prob_int = probs_int.log_prob(action_int)
        log_prob_ext = probs_ext.log_prob(action_ext)
        log_prob_edge_1 = probs_edge_1.log_prob(action_edge_1)
        log_prob_edge_2 = probs_edge_2.log_prob(action_edge_2)
        log_prob_write_value = probs_write_value.log_prob(action_write_value)
        
        entropy_int = probs_int.entropy()
        entropy_ext = probs_ext.entropy()
        entropy_edge_1 = probs_edge_1.entropy()
        entropy_edge_2 = probs_edge_2.entropy()
        entropy_write_value = probs_write_value.entropy()

        return torch.stack([
            log_prob_int, log_prob_ext, log_prob_edge_1, log_prob_edge_2, log_prob_write_value
        ], dim=-1), torch.stack([
            entropy_int, entropy_ext, entropy_edge_1, entropy_edge_2, entropy_write_value
        ], dim=-1)


    def get_log_probability_with_aux_loss(self, context, selected_action, valid_actions=None):
        # now context has shape (batch, context_size + 1, self.packed_context_size)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.long).to(self.device)

        context_full = context
        context = context_full[:, :-1, :]  # remove last context for computing logprob

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
            available_flags = valid_actions[:, :, :self.int_action_size].to(self.device)
            available_actions = valid_actions[:, :, self.int_action_size:].to(self.device)

        logits_int, logits_ext, edge_1_logits, edge_2_logits, write_value_logits, _ = self.compute(context)

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)
        probs_edge_1 = Categorical(logits=edge_1_logits)
        probs_edge_2 = Categorical(logits=edge_2_logits)
        probs_write_value = Categorical(logits=write_value_logits)

        action_int = selected_action[:, :, 0]
        action_ext = selected_action[:, :, 1]
        action_edge_1 = selected_action[:, :, 1 + 1 + 0]
        action_edge_2 = selected_action[:, :, 1 + 1 + 1]
        action_write_value = selected_action[:, :, 1 + 1 + 2]

        log_prob_int = probs_int.log_prob(action_int)
        log_prob_ext = probs_ext.log_prob(action_ext)
        log_prob_edge_1 = probs_edge_1.log_prob(action_edge_1)
        log_prob_edge_2 = probs_edge_2.log_prob(action_edge_2)
        log_prob_write_value = probs_write_value.log_prob(action_write_value)
        
        entropy_int = probs_int.entropy()
        entropy_ext = probs_ext.entropy()
        entropy_edge_1 = probs_edge_1.entropy()
        entropy_edge_2 = probs_edge_2.entropy()
        entropy_write_value = probs_write_value.entropy()

        return torch.stack([
            log_prob_int, log_prob_ext, log_prob_edge_1, log_prob_edge_2, log_prob_write_value
        ], dim=-1), torch.stack([
            entropy_int, entropy_ext, entropy_edge_1, entropy_edge_2, entropy_write_value
        ], dim=-1), None # No auxiliary loss for flipflop task


    def get_log_probability_with_value(self, context, selected_action, valid_actions=None):
        # now context has shape (batch, context_size + 1, self.packed_context_size)
        # Return log prob will only have context_size, but value will have context_size + 1

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.long).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
            available_flags = valid_actions[:, :, :self.int_action_size].to(self.device)
            available_actions = valid_actions[:, :, self.int_action_size:].to(self.device)

        logits_int, logits_ext, edge_1_logits, edge_2_logits, write_value_logits, values = self.compute(context)

        # remove last context for computing logprob and value, since it corresponds to the next observation after taking the last action
        logits_int = logits_int[:, :-1, :]
        logits_ext = logits_ext[:, :-1, :]
        edge_1_logits = edge_1_logits[:, :-1, :]
        edge_2_logits = edge_2_logits[:, :-1, :]
        write_value_logits = write_value_logits[:, :-1, :]
        # but maintain values for all contexts, since we need the bootstrap value for the last context

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)
        probs_edge_1 = Categorical(logits=edge_1_logits)
        probs_edge_2 = Categorical(logits=edge_2_logits)
        probs_write_value = Categorical(logits=write_value_logits)

        action_int = selected_action[:, :, 0]
        action_ext = selected_action[:, :, 1]
        action_edge_1 = selected_action[:, :, 1 + 1 + 0]
        action_edge_2 = selected_action[:, :, 1 + 1 + 1]
        action_write_value = selected_action[:, :, 1 + 1 + 2]

        log_prob_int = probs_int.log_prob(action_int)
        log_prob_ext = probs_ext.log_prob(action_ext)
        log_prob_edge_1 = probs_edge_1.log_prob(action_edge_1)
        log_prob_edge_2 = probs_edge_2.log_prob(action_edge_2)
        log_prob_write_value = probs_write_value.log_prob(action_write_value)
        
        entropy_int = probs_int.entropy()
        entropy_ext = probs_ext.entropy()
        entropy_edge_1 = probs_edge_1.entropy()
        entropy_edge_2 = probs_edge_2.entropy()
        entropy_write_value = probs_write_value.entropy()

        return torch.stack([
            log_prob_int, log_prob_ext, log_prob_edge_1, log_prob_edge_2, log_prob_write_value
        ], dim=-1), torch.stack([
            entropy_int, entropy_ext, entropy_edge_1, entropy_edge_2, entropy_write_value
        ], dim=-1), values


class Projector(Base_Projector):
    
    def get_log_probability_with_value(self, context, selected_action, valid_actions=None):
        all_logprobs, all_entropy, values = self.master_core.get_log_probability_with_value(context, selected_action, valid_actions)
        log_probs = all_logprobs[:, :, self.selected_indices].sum(dim=-1)  # sum over selected logprob components
        entropy = all_entropy[:, :, self.selected_indices].sum(dim=-1)
        return log_probs, entropy, values
    