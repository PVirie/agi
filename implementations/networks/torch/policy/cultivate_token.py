import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np
import logging

from implementations.networks.torch.components.base import init_weights
from implementations.networks.torch.components.base import Categorical_With_Mask
from implementations.networks.torch.components.std_resnet import ResNet
from implementations.networks.torch.components.std_conv import ImpalaCNN
from implementations.networks.torch.policy.base_token import Policy_Core as Base_Policy_Core
from utilities.safe_torch_module import Safe_nn_Module


class Policy_Core(Base_Policy_Core):

    def __init__(self, 
                 int_action_size, ext_action_size, 
                 goal_size, inventory_size,
                 dict_size, embedding_dim,
                 width, height, channel,
                 hidden_size, layers, 
                 history_steps=0, max_temporal_len=32, 
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="base_token_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.width = width
        self.height = height
        self.channel = channel

        self.goal_size = goal_size
        self.inventory_size = inventory_size
        self.obs_size = width * height * channel
        self.hidden_size = hidden_size

        self.int_action_size = int_action_size  # num classes for flag
        self.ext_action_size = ext_action_size
        self.position_size = self.goal_size # use position to store sub-goal information
        self.content_size = self.goal_size + self.inventory_size + self.obs_size
        self.packed_action_size = 1 + 1 + self.position_size + self.content_size
        self.packed_context_size = 1 + 1 + 1 + self.position_size + self.content_size

        self.goal_embedding = nn.Embedding(dict_size, embedding_dim)  # for goal token
        self.image_embedding = nn.Embedding(256, 4)  # for image pixels, shared across channels
        self.feature_channel = self.channel * 4
        self.conv_layers = ImpalaCNN(
            output_dims=hidden_size, 
            input_channels=self.feature_channel, width=width, height=height,
            depths=[16, 32, 32]
        )

        predict_input_dim = goal_size * embedding_dim + inventory_size * embedding_dim + hidden_size  # goal, inv, obs -> hidden
        self.backbone = ResNet(
            output_dims=hidden_size, 
            input_dims=predict_input_dim, 
            hidden_dims=hidden_size, 
            layers=layers
        )

        self.evaluator = ResNet(
            output_dims=inventory_size * embedding_dim + hidden_size, 
            input_dims=inventory_size * embedding_dim + hidden_size, 
            hidden_dims=hidden_size, layers=layers
        ) # inv, obs -> embedding + hidden

        # hidden -> action
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

        # hidden -> sub goal
        self.head_subgoal = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, goal_size * embedding_dim)   # predict next subgoal in embedded space
        )

        self.reset_parameters()
        self.load(override_persistence_path=first_load_path)
        self.eval()


    def reset_parameters(self):
        # Reset parameters of all layers

        self.goal_embedding.reset_parameters()
        self.image_embedding.reset_parameters()
        self.conv_layers.reset_parameters()

        self.backbone.reset_parameters()
        self.evaluator.reset_parameters()

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
        self.head_subgoal.apply(init_actor_weights)


    def compute(self, context):
        # context has shape (batch, context_size, self.packed_context_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        reward = context[:, :, 0:1]  # (batch_size, context_size, 1)
        flag_onehot = torch.nn.functional.one_hot(context[:, :, 1].long(), num_classes=self.int_action_size).float()
        action_onehot = torch.nn.functional.one_hot(context[:, :, 2].long(), num_classes=self.ext_action_size).float()
        last_subgoal = context[:, :, (1 + 1 + 1): (1 + 1 + 1 + self.position_size)]  # (batch_size, context_size, goal_size)
        goal = context[:, :, (1 + 1 + 1 + self.position_size): (1 + 1 + 1 + self.position_size + self.goal_size)]  # (batch_size, context_size, goal_size)
        inv = context[:, :, (1 + 1 + 1 + self.position_size + self.goal_size): (1 + 1 + 1 + self.position_size + self.goal_size + self.inventory_size)]  # (batch_size, context_size, inventory_size)
        obs = context[:, :, (1 + 1 + 1 + self.position_size + self.goal_size + self.inventory_size): ]  # (batch_size, context_size, obs_size)

        last_subgoal_embedded = self.goal_embedding(last_subgoal.long())  # (batch_size, context_size, goal_size, embedding_dim)
        last_subgoal_embedded = last_subgoal_embedded.view(batch_size, context_size, -1)  # (batch_size, context_size, goal_size * embedding_dim)
        goal_embedded = self.goal_embedding(goal.long())  # (batch_size, context_size, goal_size, embedding_dim)
        goal_embedded = goal_embedded.view(batch_size, context_size, -1)  # (batch_size, context_size, goal_size * embedding_dim)
        inv_embedded = self.goal_embedding(inv.long())  # (batch_size, context_size, inventory_size, inv_size * embedding_dim)
        inv_embedded = inv_embedded.view(batch_size, context_size, -1)  # (batch_size, context_size, inventory_size * embedding_dim)
        
        obs_embedded = self.image_embedding(obs.long())  # (batch_size, context_size, obs_size, embedding_dim)
        obs_features = torch.reshape(obs_embedded, (batch_size * context_size, self.height, self.width, self.feature_channel))  # (batch_size * context_size, height, width, channel * embedding_dim)
        obs_features = obs_features.permute(0, 3, 1, 2)  # (batch_size * context_size, channel * embedding_dim, height, width)
        obs_features = self.conv_layers(obs_features)  # (batch_size * context_size, hidden_size)
        obs_features = obs_features.view(batch_size, context_size, self.hidden_size) # (batch_size, context_size, hidden_size)

        # Base flow

        base_input = torch.concat([goal_embedded, inv_embedded, obs_features], dim=-1)  # (batch_size, context_size, vec_dim)
        base_output = self.backbone(base_input)  # (batch_size, context_size, hidden_size)

        int_logits = self.head_int(base_output)  # (batch_size, context_size, int_action_size)
        ext_logits = self.head_ext(base_output)  # (batch_size, context_size, ext_action_size)
        
        # Cultivate flow

        lw_eval_input = torch.concat([inv_embedded, obs_features], dim=-1)  # (batch_size, context_size, inventory_size * embedding_dim + hidden_size)
        lw_result = self.evaluator(lw_eval_input)  # (batch_size, context_size, inventory_size * embedding_dim + hidden_size)

        sub_input = torch.concat([goal_embedded, lw_result], dim=-1)  # (batch_size, context_size, vec_dim)
        sub_output = self.backbone(sub_input)  # (batch_size, context_size, hidden_size)

        subgoal_logits = self.head_subgoal(sub_output)  # (batch_size, context_size, goal_size * embedding_dim)

        lwlv_input = torch.concat([subgoal_logits, inv_embedded, obs_features], dim=-1)  # (batch_size, context_size, vec_dim)
        lwlv_output = self.backbone(lwlv_input)  # (batch_size, context_size, hidden_size)

        aux_int_logits = self.head_int(lwlv_output)  # (batch_size, context_size, int_action_size)
        aux_ext_logits = self.head_ext(lwlv_output)  # (batch_size, context_size, ext_action_size)

        # calculate aux loss
        action_loss = torch.mean((int_logits - aux_int_logits)**2, dim=-1) + torch.mean((ext_logits - aux_ext_logits)**2, dim=-1)
        # subgoal_logits should not deviate much from last_subgoal_embedded 
        subgoal_loss = torch.mean((subgoal_logits - last_subgoal_embedded)**2, dim=-1)
        aux_loss = action_loss + 0.2 * subgoal_loss

        return int_logits, ext_logits, subgoal_logits, aux_loss
    

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

        logits_int, logits_ext, position_logits, aux_loss = self.compute(context)

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)

        batch_size = context.size(0)
        context_size = context.size(1)

        action_int = probs_int.sample()
        action_ext = probs_ext.sample()
        next_position = torch.argmax(position_logits.view(batch_size, context_size, self.goal_size, -1), dim=-1)  # (batch_size, context_size, goal_size)
        action_content = np.zeros((batch_size, context_size, self.content_size), dtype=np.float32)

        action = np.concatenate([
            action_int.unsqueeze(-1).cpu().numpy(),
            action_ext.unsqueeze(-1).cpu().numpy(),
            next_position.cpu().numpy(),
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

        logits_int, logits_ext, position_logits, aux_loss = self.compute(context)

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)

        action_int = selected_action[:, :, 0]
        action_ext = selected_action[:, :, 1]

        log_prob_int = probs_int.log_prob(action_int)
        log_prob_ext = probs_ext.log_prob(action_ext)
        
        entropy_int = probs_int.entropy()
        entropy_ext = probs_ext.entropy()

        return torch.stack([
            log_prob_int, log_prob_ext
        ], dim=-1), torch.stack([
            entropy_int, entropy_ext
        ], dim=-1)
    

    def get_log_probability_with_aux_loss(self, context, selected_action, valid_actions=None):
        # now context has shape (batch, context_size + 1, self.packed_context_size)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

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

        logits_int, logits_ext, position_logits, aux_loss = self.compute(context)

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)

        action_int = selected_action[:, :, 0]
        action_ext = selected_action[:, :, 1]

        log_prob_int = probs_int.log_prob(action_int)
        log_prob_ext = probs_ext.log_prob(action_ext)
        
        entropy_int = probs_int.entropy()
        entropy_ext = probs_ext.entropy()

        return torch.stack([
            log_prob_int, log_prob_ext
        ], dim=-1), torch.stack([
            entropy_int, entropy_ext
        ], dim=-1), aux_loss  # no aux loss for now



