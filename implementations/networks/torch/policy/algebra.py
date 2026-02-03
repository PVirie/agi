import torch
import torch.nn as nn
import numpy as np
import logging

from implementations.networks.torch.components.base import init_weights
from implementations.networks.torch.components.base import Categorical_With_Mask
from implementations.networks.torch.components.temporal_unet import TemporalUNet
from implementations.networks.torch.algebra_core.core import Algebra_Core
from implementations.networks.torch.policy.arcagi3 import Policy_Core as ARCAGI3_Policy_Core
from utilities.safe_torch_module import Safe_nn_Module


class Policy_Core(ARCAGI3_Policy_Core):

    def __init__(self, action_size, position_size, width, height, channel, hidden_size, layers, history_steps=0, max_temporal_len=32, device=None, persistence_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="algebra_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.flag_size = 6  # num classes for flag
        self.action_size = action_size
        self.position_size = position_size
        self.content_size = channel * width * height
        self.packed_action_size = 1 + 3 + position_size + self.content_size  # int_flag + action + x + y + position + content
        self.packed_context_size = 1 + 1 + 3 + position_size + self.content_size  # reward + packed_action_size

        self.width = width
        self.height = height
        self.channel = channel
        self.hidden_size = hidden_size
        self.position_features = 8

        vec_dim = action_size + width + height
        self.temporal_unet = TemporalUNet(
            input_channels=channel, width=width, height=height,
            vec_dim=vec_dim, hidden_dim=hidden_size * self.position_features,
            depths=layers, history_steps=0, max_temporal_len=max_temporal_len)
        
        self.algebra_core = Algebra_Core(
            position_output_dim=position_size // self.position_features,
            num_algebras=32,
            context_size=history_steps
        )

        self.position_step = nn.Sequential(
            nn.Linear(position_size + vec_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, position_size)
        )

        self.position_project = nn.Sequential(
            nn.Linear(position_size + position_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.feature_project = nn.Sequential(
            nn.Linear(hidden_size * self.position_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
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


    def reset_parameters(self):
        # Reset parameters of all layers
        self.temporal_unet.reset_parameters()
        self.algebra_core.apply(init_weights)
        self.position_step.apply(init_weights)

        def init_actor_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        last_position = context[:, :, (1 + 1 + 3): (1 + 1 + 3 + self.position_size)]  # (batch_size, context_size, position_size)

        # make one hot encoding for action, location
        reward = reward_action_part[:, :, 0:1]  # (batch_size, context_size, 1)
        flag_onehot = torch.nn.functional.one_hot(reward_action_part[:, :, 1].long(), num_classes=self.flag_size).float()
        action_onehot = torch.nn.functional.one_hot(reward_action_part[:, :, 2].long(), num_classes=self.action_size).float()
        x_onehot = torch.nn.functional.one_hot(reward_action_part[:, :, 3].long(), num_classes=self.width).float()
        y_onehot = torch.nn.functional.one_hot(reward_action_part[:, :, 4].long(), num_classes=self.height).float()
        
        action_part = torch.concat([action_onehot, x_onehot, y_onehot], dim=-1)  # (batch_size, context_size, 1 + 1 + 3 + position_size)
        features, x_logits, y_logits, content_logits = self.temporal_unet(image_part, action_part) 
        # features has shape (batch_size, context_size, hidden_size * position_features)

        # now convert features to position features
        # first merge batch with position_features
        split_features = torch.reshape(features, (batch_size, context_size, self.position_features, self.hidden_size))
        position_output = self.algebra_core(split_features)  # (batch_size, context_size, position_features, position_size / position_features)
        position_output = torch.reshape(position_output, (batch_size, context_size, self.position_size))

        # get next position from step
        next_position = self.position_step(torch.concat([last_position, action_part], dim=-1))

        projected_position = self.position_project(torch.concat([position_output, next_position], dim=-1)) # (batch_size, context_size, hidden_size)
        projected_features = self.feature_project(features) # (batch_size, context_size, hidden_size)

        # compute dropout
        if self.training:
            keep_prob = 0.75
            mask = torch.empty([batch_size, context_size, 1], device=self.device).bernoulli_(keep_prob)
            merged_features = projected_position + projected_features * mask / keep_prob
        else:
            merged_features = projected_position + projected_features

        logits_flag = self.head_flag(merged_features)    # (B, T, flag_size)
        logits_action = self.head_action(merged_features) # (B, T, action_size)
        content_logits = self.head_content(torch.reshape(content_logits, (batch_size * context_size, self.channel, self.height, self.width)))

        content_logits = torch.reshape(content_logits, (batch_size, context_size, self.content_size))

        return logits_flag, logits_action, x_logits, y_logits, next_position, content_logits