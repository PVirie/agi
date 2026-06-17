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
from implementations.networks.torch.components.transformer import InstructionTransformer, get_padding_mask
from implementations.networks.torch.policy.base import Policy_Core as Base_Policy_Core
from implementations.networks.torch.policy.base import Projector as Base_Projector
from utilities.safe_torch_module import Safe_nn_Module


class Policy_Core(Base_Policy_Core, Policy_Value_Network):

    def __init__(self, 
                 int_action_size, ext_action_size, 
                 position_size,
                 content_size,
                 dict_size, embedding_dim, pad_token_id,
                 hidden_size, layers, 
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="flipflop_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim

        self.int_action_size = int_action_size  # num classes for flag
        self.ext_action_size = ext_action_size
        self.position_size = position_size
        self.content_size = content_size
        self.packed_action_size = 1 + 1 + self.position_size + self.content_size
        self.packed_context_size = 1 + 1 + 1 + self.position_size + self.content_size

        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(dict_size, embedding_dim, padding_idx=pad_token_id)  # for goal tokens
        self.feature_extraction = InstructionTransformer(
            input_dim=embedding_dim,
            d_model=hidden_size,
            nhead=8, 
            num_layers=2, 
            max_len=content_size
        )

        # goal, inv, obs -> hidden
        self.backbone = ResNet(
            output_dims=hidden_size, 
            input_dims=hidden_size, 
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
            nn.Linear(hidden_size, content_size - 1)
        )
        self.head_edge_2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, content_size - 1)
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
        self.embedding.reset_parameters()
        self.feature_extraction.reset_parameters()
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

        reward = context[:, :, 0:1]  # (batch_size, context_size, 1)
        flag_onehot = torch.nn.functional.one_hot(context[:, :, 1].long(), num_classes=self.int_action_size).float()
        action_onehot = torch.nn.functional.one_hot(context[:, :, 2].long(), num_classes=self.ext_action_size).float()
        content = context[:, :, 1 + 1 + 1 + self.position_size: ]  # (batch_size, context_size, content_size)
                          
        padding_mask = get_padding_mask(content, self.pad_token_id)  # (batch_size, context_size, goal_size)
        embedded = self.embedding(content.long())  # (batch_size, context_size, goal_size, embedding_dim)
        features = self.feature_extraction(embedded, padding_mask)  # (batch_size, context_size, hidden_size)
        
        # Base flow

        base_input = features  # (batch_size, context_size, hidden_size)
        base_output = self.backbone(base_input)  # (batch_size, context_size, hidden_size)

        int_logits = self.head_int(base_output)  # (batch_size, context_size, int_action_size)
        ext_logits = self.head_ext(base_output)  # (batch_size, context_size, ext_action_size)
        edge_1_logits = self.head_edge_1(base_output)  # (batch_size, context_size, content_size - 1)
        edge_2_logits = self.head_edge_2(base_output)  # (batch_size, context_size, content_size - 1)

        values = self.head_value(base_output)  # (batch_size, context_size, 1)
        values = values.squeeze(-1)  # (batch_size, context_size)

        return int_logits, ext_logits, edge_1_logits, edge_2_logits, values
    

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

        logits_int, logits_ext, edge_1_logits, edge_2_logits, values = self.compute(context)

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)
        probs_edge_1 = Categorical(logits=edge_1_logits)
        probs_edge_2 = Categorical(logits=edge_2_logits)

        batch_size = context.size(0)
        context_size = context.size(1)

        action_int = probs_int.sample()
        action_ext = probs_ext.sample()
        edge_1 = probs_edge_1.sample()
        edge_2 = probs_edge_2.sample()
        action_content = np.zeros((batch_size, context_size, self.content_size), dtype=np.float32)

        action = np.concatenate([
            action_int.unsqueeze(-1).cpu().numpy(),
            action_ext.unsqueeze(-1).cpu().numpy(),
            edge_1.unsqueeze(-1).cpu().numpy(),
            edge_2.unsqueeze(-1).cpu().numpy(),
            action_content
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

        logits_int, logits_ext, edge_1_logits, edge_2_logits, values = self.compute(context)

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)
        probs_edge_1 = Categorical(logits=edge_1_logits)
        probs_edge_2 = Categorical(logits=edge_2_logits)

        action_int = selected_action[:, :, 0]
        action_ext = selected_action[:, :, 1]
        action_edge_1 = selected_action[:, :, 1 + 1 + 0]
        action_edge_2 = selected_action[:, :, 1 + 1 + 1]

        log_prob_int = probs_int.log_prob(action_int)
        log_prob_ext = probs_ext.log_prob(action_ext)
        log_prob_edge_1 = probs_edge_1.log_prob(action_edge_1)
        log_prob_edge_2 = probs_edge_2.log_prob(action_edge_2)
        
        entropy_int = probs_int.entropy()
        entropy_ext = probs_ext.entropy()
        entropy_edge_1 = probs_edge_1.entropy()
        entropy_edge_2 = probs_edge_2.entropy()

        return torch.stack([
            log_prob_int, log_prob_ext, log_prob_edge_1, log_prob_edge_2
        ], dim=-1), torch.stack([
            entropy_int, entropy_ext, entropy_edge_1, entropy_edge_2
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

        logits_int, logits_ext, edge_1_logits, edge_2_logits, values = self.compute(context)

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)
        probs_edge_1 = Categorical(logits=edge_1_logits)
        probs_edge_2 = Categorical(logits=edge_2_logits)

        action_int = selected_action[:, :, 0]
        action_ext = selected_action[:, :, 1]
        action_edge_1 = selected_action[:, :, 1 + 1 + 0]
        action_edge_2 = selected_action[:, :, 1 + 1 + 1]

        log_prob_int = probs_int.log_prob(action_int)
        log_prob_ext = probs_ext.log_prob(action_ext)
        log_prob_edge_1 = probs_edge_1.log_prob(action_edge_1)
        log_prob_edge_2 = probs_edge_2.log_prob(action_edge_2)
        
        entropy_int = probs_int.entropy()
        entropy_ext = probs_ext.entropy()
        entropy_edge_1 = probs_edge_1.entropy()
        entropy_edge_2 = probs_edge_2.entropy()

        return torch.stack([
            log_prob_int, log_prob_ext, log_prob_edge_1, log_prob_edge_2
        ], dim=-1), torch.stack([
            entropy_int, entropy_ext, entropy_edge_1, entropy_edge_2
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

        logits_int, logits_ext, edge_1_logits, edge_2_logits, values = self.compute(context)

        # remove last context for computing logprob and value, since it corresponds to the next observation after taking the last action
        logits_int = logits_int[:, :-1, :]
        logits_ext = logits_ext[:, :-1, :]
        edge_1_logits = edge_1_logits[:, :-1, :]
        edge_2_logits = edge_2_logits[:, :-1, :]
        # but maintain values for all contexts, since we need the bootstrap value for the last context

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)
        probs_edge_1 = Categorical(logits=edge_1_logits)
        probs_edge_2 = Categorical(logits=edge_2_logits)

        action_int = selected_action[:, :, 0]
        action_ext = selected_action[:, :, 1]
        action_edge_1 = selected_action[:, :, 1 + 1 + 0]
        action_edge_2 = selected_action[:, :, 1 + 1 + 1]

        log_prob_int = probs_int.log_prob(action_int)
        log_prob_ext = probs_ext.log_prob(action_ext)
        log_prob_edge_1 = probs_edge_1.log_prob(action_edge_1)
        log_prob_edge_2 = probs_edge_2.log_prob(action_edge_2)
        
        entropy_int = probs_int.entropy()
        entropy_ext = probs_ext.entropy()
        entropy_edge_1 = probs_edge_1.entropy()
        entropy_edge_2 = probs_edge_2.entropy()

        return torch.stack([
            log_prob_int, log_prob_ext, log_prob_edge_1, log_prob_edge_2
        ], dim=-1), torch.stack([
            entropy_int, entropy_ext, entropy_edge_1, entropy_edge_2
        ], dim=-1), values


class Projector(Base_Projector):
    
    def get_log_probability_with_value(self, context, selected_action, valid_actions=None):
        all_logprobs, all_entropy, values = self.master_core.get_log_probability_with_value(context, selected_action, valid_actions)
        log_probs = all_logprobs[:, :, self.selected_indices].sum(dim=-1)  # sum over selected logprob components
        entropy = all_entropy[:, :, self.selected_indices].sum(dim=-1)
        return log_probs, entropy, values
    