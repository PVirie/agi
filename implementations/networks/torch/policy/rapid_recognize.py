import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np
import logging

from implementations.networks.torch.components.base import init_weights
from implementations.networks.torch.components.base import Categorical_With_Mask
from implementations.networks.torch.components.temporal_unet import TemporalUNet
from implementations.networks.torch.policy.arcagi3 import Policy_Core as ARCAGI3_Policy_Core, Projector as ARCAGI3_Projector
from utilities.safe_torch_module import Safe_nn_Module


class Policy_Core(ARCAGI3_Policy_Core):

    def __init__(self, action_size, position_slot, width, height, channel, hidden_size, layers, history_steps=0, max_temporal_len=32, device=None, persistence_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="rapid_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.flag_size = 6  # num classes for flag
        self.action_size = action_size
        self.position_slot = position_slot
        self.position_size = 1
        self.content_size = channel * width * height
        self.packed_action_size = 1 + 3 + self.position_size + self.content_size  # int_flag + action + x + y + position + content
        self.packed_context_size = 1 + 1 + 3 + self.position_size + self.content_size  # reward + packed_action_size

        self.width = width
        self.height = height
        self.channel = channel
        self.hidden_size = hidden_size

        vec_dim = self.action_size
        self.temporal_unet = TemporalUNet(
            output_dims=hidden_size,
            input_channels=channel, width=width, height=height,
            vec_dim=vec_dim, hidden_dim=hidden_size,
            depths=layers, history_steps=history_steps, max_temporal_len=max_temporal_len)
        
        self.recognize_module = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, position_slot)
        )

        self.propagator = nn.Sequential(
            nn.Linear(position_slot, position_slot)
        )

        self.position_module = nn.Sequential(
            nn.Linear(position_slot, hidden_size)
        )

        self.head_flag = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.flag_size)   # self.flag_size classes
        )
        self.head_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_size)   # action_size classes
        )
        self.head_content = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1)
        )

        self.content_logstd = nn.Parameter(torch.zeros(1, 1, self.content_size))

        self.reset_parameters()
        self.load()
        self.eval()


    def slow_parameters(self):
        # override to return all parameters except the recognize_module parameters
        params = []
        for name, param in self.named_parameters():
            if "recognize_module" not in name:
                params.append(param)
        return params


    def fast_parameters(self):
        # override to return only the recognize_module parameters
        params = []
        for name, param in self.named_parameters():
            if "recognize_module" in name:
                params.append(param)
        return params


    def reset_parameters(self):
        # Reset parameters of all layers
        self.temporal_unet.reset_parameters()
        self.propagator.apply(init_weights)
        self.position_module.apply(init_weights)

        def init_actor_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.recognize_module.apply(init_actor_weights)
        self.head_flag.apply(init_actor_weights)
        self.head_action.apply(init_actor_weights)
        self.head_content.apply(init_actor_weights)
        nn.init.constant_(self.content_logstd, 0.0)


    def compute(self, context):
        # context has shape (batch, context_size, self.packed_context_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        # first slice the image content
        image_content = context[:, :, (1 + 1 + 3 + self.position_size): ]  # (batch_size, context_size, content_size)
        image_part = torch.reshape(image_content, (batch_size, context_size, self.channel, self.height, self.width))
        reward_action_part = context[:, :, :(1 + 1 + 3)]  # (batch_size, context_size, 1 + 1 + 3)
        last_position = context[:, :, (1 + 1 + 3)]  # (batch_size, context_size)

        # make one hot encoding for action, location
        reward = reward_action_part[:, :, 0:1]  # (batch_size, context_size, 1)
        flag_onehot = torch.nn.functional.one_hot(reward_action_part[:, :, 1].long(), num_classes=self.flag_size).float()
        action_onehot = torch.nn.functional.one_hot(reward_action_part[:, :, 2].long(), num_classes=self.action_size).float()
        x_onehot = torch.nn.functional.one_hot(reward_action_part[:, :, 3].long(), num_classes=self.width).float()
        y_onehot = torch.nn.functional.one_hot(reward_action_part[:, :, 4].long(), num_classes=self.height).float()
        last_position_onehot = torch.nn.functional.one_hot(last_position.long(), num_classes=self.position_slot).float()

        features, x_logits, y_logits, content_logits = self.temporal_unet(image_part, action_onehot)

        # detach gradient for feature from recognize module
        recognized_position_logits = self.recognize_module(features.detach())  # (B, T, position_slot)
        propagated_position_logits = self.propagator(last_position_onehot)  # (B, T, position_slot)
        position_logits = recognized_position_logits + propagated_position_logits  # (B, T, position_slot)

        probs_positions = Categorical(logits=position_logits)
        positions = probs_positions.sample()  # (B, T)
        position_onehot = torch.nn.functional.one_hot(positions.long(), num_classes=self.position_slot).float() # (B, T, position_slot)
        position_features = self.position_module(position_onehot)  # (B, T, hidden_size)

        # compute dropout
        if self.training:
            keep_prob = 0.9
            mask = torch.empty([batch_size, context_size, 1], device=self.device).bernoulli_(keep_prob)
            merged_features = position_features + features * mask / keep_prob
        else:
            merged_features = position_features + features
        
        logits_flag = self.head_flag(merged_features)    # (B, T, flag_size)
        logits_action = self.head_action(merged_features) # (B, T, action_size)
        content_logits = self.head_content(torch.reshape(content_logits, (batch_size * context_size, self.channel, self.height, self.width)))

        content_logits = torch.reshape(content_logits, (batch_size, context_size, self.content_size))

        return logits_flag, logits_action, x_logits, y_logits, positions, probs_positions, content_logits
    

    def get_action(self, context, valid_actions=None):

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, flag_size + action_size)
            available_flags = valid_actions[:, :, :self.flag_size].to(self.device)
            available_actions = valid_actions[:, :, self.flag_size:].to(self.device)

        logits_flag, logits_action, x_logits, y_logits, positions, probs_positions, content_logits = self.compute(context)

        probs_flag = Categorical_With_Mask(logits=logits_flag, mask=available_flags)
        probs_action = Categorical_With_Mask(logits=logits_action, mask=available_actions)
        probs_x = Categorical(logits=x_logits)
        probs_y = Categorical(logits=y_logits)
        probs_content = Normal(content_logits, torch.exp(self.content_logstd.expand_as(content_logits)))

        action_flag = probs_flag.sample()
        action_action = probs_action.sample()
        action_x = probs_x.sample()
        action_y = probs_y.sample()
        action_content = probs_content.sample()

        action = torch.cat([
            action_flag.unsqueeze(-1),
            action_action.unsqueeze(-1),
            action_x.unsqueeze(-1),
            action_y.unsqueeze(-1),
            positions.unsqueeze(-1),
            action_content
        ], dim=-1)

        return action.cpu().numpy().astype(float)
    

    def get_log_probability(self, context, selected_action, valid_actions=None):
        # context has shape (batch, context_size, self.packed_context_size)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, flag_size + action_size)
            available_flags = valid_actions[:, :, :self.flag_size].to(self.device)
            available_actions = valid_actions[:, :, self.flag_size:].to(self.device)

        logits_flag, logits_action, x_logits, y_logits, positions, probs_positions, content_logits = self.compute(context)

        probs_flag = Categorical_With_Mask(logits=logits_flag, mask=available_flags)
        probs_action = Categorical_With_Mask(logits=logits_action, mask=available_actions)
        probs_x = Categorical(logits=x_logits)
        probs_y = Categorical(logits=y_logits)
        probs_content = Normal(content_logits, torch.exp(self.content_logstd.expand_as(content_logits)))

        action_flag = selected_action[:, :, 0]
        action_action = selected_action[:, :, 1]
        action_x = selected_action[:, :, 2]
        action_y = selected_action[:, :, 3]
        action_position = selected_action[:, :, 4]
        action_content = selected_action[:, :, (1 + 3 + self.position_size):]

        log_prob_flag = probs_flag.log_prob(action_flag)
        log_prob_action = probs_action.log_prob(action_action)
        log_prob_x = probs_x.log_prob(action_x)
        log_prob_y = probs_y.log_prob(action_y)
        log_prob_position = probs_positions.log_prob(action_position)
        log_prob_content = probs_content.log_prob(action_content).mean(-1)
        
        entropy_flag = probs_flag.entropy()
        entropy_action = probs_action.entropy()
        entropy_x = probs_x.entropy()
        entropy_y = probs_y.entropy()
        entropy_position = probs_positions.entropy()
        entropy_content = probs_content.entropy().mean(-1)

        return torch.stack([
            log_prob_flag, log_prob_action, log_prob_x, log_prob_y, log_prob_position, log_prob_content
        ], dim=-1), torch.stack([
            entropy_flag, entropy_action, entropy_x, entropy_y, entropy_position, entropy_content
        ], dim=-1)


    def get_log_probability_with_svl_loss(self, context, selected_action, valid_actions=None):
        # now context has shape (batch, context_size + 1, self.packed_context_size)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        context_full = context
        context = context_full[:, :-1, :]  # remove last context for computing logprob
        target_content = context_full[:, 1:, (1 + 1 + 3 + self.position_size): ]  # only content part for action loss

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, flag_size + action_size)
            available_flags = valid_actions[:, :, :self.flag_size].to(self.device)
            available_actions = valid_actions[:, :, self.flag_size:].to(self.device)

        logits_flag, logits_action, x_logits, y_logits, positions, probs_positions, content_logits = self.compute(context)

        probs_flag = Categorical_With_Mask(logits=logits_flag, mask=available_flags)
        probs_action = Categorical_With_Mask(logits=logits_action, mask=available_actions)
        probs_x = Categorical(logits=x_logits)
        probs_y = Categorical(logits=y_logits)
        probs_content = Normal(content_logits, torch.exp(self.content_logstd.expand_as(content_logits)))

        action_flag = selected_action[:, :, 0]
        action_action = selected_action[:, :, 1]
        action_x = selected_action[:, :, 2]
        action_y = selected_action[:, :, 3]
        action_position = selected_action[:, :, 4]
        action_content = selected_action[:, :, (1 + 3 + self.position_size):]

        log_prob_flag = probs_flag.log_prob(action_flag)
        log_prob_action = probs_action.log_prob(action_action)
        log_prob_x = probs_x.log_prob(action_x)
        log_prob_y = probs_y.log_prob(action_y)
        log_prob_position = probs_positions.log_prob(action_position)
        log_prob_content = probs_content.log_prob(action_content).mean(-1)
        
        entropy_flag = probs_flag.entropy()
        entropy_action = probs_action.entropy()
        entropy_x = probs_x.entropy()
        entropy_y = probs_y.entropy()
        entropy_position = probs_positions.entropy()
        entropy_content = probs_content.entropy().mean(-1)

        # svl_unsum_loss
        svl_unsum_loss = torch.mean((probs_content.mean - target_content) ** 2, dim=-1)

        return torch.stack([
            log_prob_flag, log_prob_action, log_prob_x, log_prob_y, log_prob_position, log_prob_content
        ], dim=-1), torch.stack([
            entropy_flag, entropy_action, entropy_x, entropy_y, entropy_position, entropy_content
        ], dim=-1), svl_unsum_loss



# return only selected statistics
class Projector(ARCAGI3_Projector):

    def slow_parameters(self):
        return self.master_core.slow_parameters()
    
    def fast_parameters(self):
        return self.master_core.fast_parameters()