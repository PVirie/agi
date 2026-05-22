import torch
import torch.nn as nn
import torch.nn.functional as F
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
                 goal_size,
                 width, height, channel, 
                 hidden_size, layers, 
                 hlv_steps=2, 
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="cultivate_token_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.width = width
        self.height = height
        self.channel = channel

        self.goal_size = goal_size
        self.obs_size = width * height * channel
        self.hidden_size = hidden_size
        self.hlv_steps = hlv_steps

        self.int_action_size = int_action_size  # num classes for flag
        self.ext_action_size = ext_action_size
        self.position_size = 1 + self.obs_size # position part includes step, image for last high-level
        self.content_size = goal_size + self.obs_size
        self.packed_action_size = 1 + 1 + self.position_size + self.content_size
        self.packed_context_size = 1 + 1 + 1 + self.position_size + self.content_size

        self.step_pos = 1 + 1 + 1
        self.last_hlv_obs_pos = self.step_pos + 1
        self.goal_pos = self.last_hlv_obs_pos + self.obs_size
        self.obs_pos = self.goal_pos + goal_size

        self.goal_feature_extraction = nn.Sequential(
            nn.Linear(goal_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.conv_layers = ImpalaCNN(
            output_dims=hidden_size, 
            input_channels=channel, 
            width=width, height=height,
            depths=layers
        )

        # inv, obs -> hidden_size
        self.abstractor = ResNet(
            output_dims=hidden_size, 
            input_dims=hidden_size, 
            hidden_dims=hidden_size, 
            layers=[hidden_size for _ in layers]
        )

        # goal, obs -> hidden
        self.frontal_lobe = ResNet(
            output_dims=hidden_size,
            input_dims=hidden_size + hidden_size,
            hidden_dims=hidden_size,
            layers=[hidden_size for _ in layers]
        )

        # goal, obs -> hidden
        self.backbone = ResNet(
            output_dims=hidden_size, 
            input_dims=hidden_size + hidden_size, 
            hidden_dims=hidden_size, 
            layers=[hidden_size for _ in layers]
        )

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
            nn.Linear(hidden_size, hidden_size)   # predict next subgoal in embedded space
        )

        self.reset_parameters()
        self.load(override_persistence_path=first_load_path)
        self.eval()


    def reset_parameters(self):
        # Reset parameters of all layers

        self.goal_feature_extraction.apply(init_weights)
        self.conv_layers.reset_parameters()
        self.abstractor.reset_parameters()

        self.frontal_lobe.reset_parameters()
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
        self.head_subgoal.apply(init_actor_weights)


    def compute(self, context):
        # context has shape (batch, context_size, self.packed_context_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        reward = context[:, :, 0:1]  # (batch_size, context_size, 1)
        flag_onehot = torch.nn.functional.one_hot(context[:, :, 1].long(), num_classes=self.int_action_size).float()
        action_onehot = torch.nn.functional.one_hot(context[:, :, 2].long(), num_classes=self.ext_action_size).float()
        last_step = context[:, :, self.step_pos: (self.step_pos + 1)].long()  # (batch_size, context_size, 1)
        last_hlv_obs = context[:, :, self.last_hlv_obs_pos: (self.last_hlv_obs_pos + self.obs_size)]  # (batch_size, context_size, obs_size)
        goal = context[:, :, self.goal_pos: (self.goal_pos + self.goal_size)]  # (batch_size, context_size, goal_size)
        obs = context[:, :, self.obs_pos: ]  # (batch_size, context_size, obs_size)

        last_hlv_obs_features = torch.reshape(last_hlv_obs, (batch_size * context_size, self.channel, self.height, self.width))
        last_hlv_obs_features = self.conv_layers(last_hlv_obs_features)  # (batch_size * context_size, hidden_size)
        last_hlv_obs_features = torch.reshape(last_hlv_obs_features, (batch_size, context_size, self.hidden_size))  # (batch_size, context_size, hidden_size)

        goal_features = self.goal_feature_extraction(goal)  # (batch_size, context_size, hidden_size)
        
        obs_features = torch.reshape(obs, (batch_size * context_size, self.channel, self.height, self.width))
        obs_features = self.conv_layers(obs_features)  # (batch_size * context_size, hidden_size)
        obs_features = torch.reshape(obs_features, (batch_size, context_size, self.hidden_size))  # (batch_size, context_size, hidden_size)

        # Cultivate flow
        last_hlv_obs_features = self.abstractor(last_hlv_obs_features)  # (batch_size, context_size, hidden)
        sub_input = torch.concat([goal_features, last_hlv_obs_features], dim=-1)  # (batch_size, context_size, vec_dim)
        sub_output = self.frontal_lobe(sub_input)  # (batch_size, context_size, hidden_size)

        midway_logits = self.head_subgoal(sub_output)  # (batch_size, context_size, hidden_size)

        obs_features = self.abstractor(obs_features)  # (batch_size, context_size, hidden)
        lwlv_input = torch.concat([midway_logits, obs_features], dim=-1)  # (batch_size, context_size, vec_dim)
        lwlv_output = self.backbone(lwlv_input)  # (batch_size, context_size, hidden_size)

        int_logits = self.head_int(lwlv_output)  # (batch_size, context_size, int_action_size)
        ext_logits = self.head_ext(lwlv_output)  # (batch_size, context_size, ext_action_size)
        
        # now calculate next step, just + 1 last_step unless it's hlv_steps - 1, then next step is 0
        next_step = torch.where(last_step >= (self.hlv_steps - 1), torch.zeros_like(last_step).long(), last_step + 1)

        return int_logits, ext_logits, next_step
    

    def get_action(self, context, valid_actions=None):

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
            available_flags = valid_actions[:, :, :self.int_action_size].to(self.device)
            available_actions = valid_actions[:, :, self.int_action_size:].to(self.device)

        logits_int, logits_ext, next_step = self.compute(context)

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)

        batch_size = context.size(0)
        context_size = context.size(1)

        # Choose next position based on nu
        selected_hlv = torch.where(next_step == 0,
                                 context[:, :, self.obs_pos:],
                                 context[:, :, self.last_hlv_obs_pos: (self.last_hlv_obs_pos + self.obs_size)])  # (batch_size, context_size, obs_size)

        action_int = probs_int.sample()
        action_ext = probs_ext.sample()
        next_position = torch.concat([next_step, selected_hlv], dim=-1)  # (batch_size, context_size, position_size)
        action_content = np.zeros((batch_size, context_size, self.content_size), dtype=np.float32)

        action = np.concatenate([
            action_int.unsqueeze(-1).cpu().numpy(),
            action_ext.unsqueeze(-1).cpu().numpy(),
            next_position.detach().cpu().numpy(),
            action_content
        ], axis=-1)

        return action.astype(float)
    

    def get_log_probability(self, context, selected_action, valid_actions=None):
        # context has shape (batch, context_size, self.packed_context_size)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
            available_flags = valid_actions[:, :, :self.int_action_size].to(self.device)
            available_actions = valid_actions[:, :, self.int_action_size:].to(self.device)

        logits_int, logits_ext, next_step = self.compute(context)

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

        logits_int, logits_ext, next_step = self.compute(context)

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
        ], dim=-1), None

