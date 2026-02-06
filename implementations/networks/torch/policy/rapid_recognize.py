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
from implementations.networks.torch.policy.arcagi3 import Policy_Core as ARCAGI3_Policy_Core
from utilities.safe_torch_module import Safe_nn_Module


class Policy_Core(ARCAGI3_Policy_Core):

    def __init__(self, action_size, position_size, width, height, channel, hidden_size, layers, history_steps=0, max_temporal_len=32, device=None, persistence_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="rapid_core", device=device, persistence_path=persistence_path)
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

        vec_dim = self.action_size
        self.temporal_unet = TemporalUNet(
            input_channels=channel, width=width, height=height,
            vec_dim=vec_dim, hidden_dim=hidden_size,
            depths=layers, history_steps=history_steps, max_temporal_len=max_temporal_len)
        
        self.recognize_module = nn.Sequential(
            nn.Linear(self.temporal_unet.out_features, position_size),
        )

        self.propagator = nn.Sequential(
            nn.Linear(position_size, position_size)
        )

        self.position_module = nn.Sequential(
            nn.Linear(position_size, hidden_size)
        )

        self.translator = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
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
        self.propagator.apply(init_weights)
        self.position_module.apply(init_weights)
        self.translator.apply(init_weights)

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
        last_position = context[:, :, (1 + 1 + 3): (1 + 1 + 3 + self.position_size)]  # (batch_size, context_size, position_size)

        # make one hot encoding for action, location
        reward = reward_action_part[:, :, 0:1]  # (batch_size, context_size, 1)
        flag_onehot = torch.nn.functional.one_hot(reward_action_part[:, :, 1].long(), num_classes=self.flag_size).float()
        action_onehot = torch.nn.functional.one_hot(reward_action_part[:, :, 2].long(), num_classes=self.action_size).float()
        x_onehot = torch.nn.functional.one_hot(reward_action_part[:, :, 3].long(), num_classes=self.width).float()
        y_onehot = torch.nn.functional.one_hot(reward_action_part[:, :, 4].long(), num_classes=self.height).float()
        
        features, x_logits, y_logits, content_logits = self.temporal_unet(image_part, action_onehot)

        recognized_position_logits = self.recognize_module(features)  # (B, T, position_size)
        propagated_position_logits = self.propagator(last_position)  # (B, T, position_size)
        position_logits = recognized_position_logits + propagated_position_logits  # (B, T, position_size)

        positions = Categorical(logits=position_logits).sample()  # (B, T)
        position_onehot = torch.nn.functional.one_hot(positions.long(), num_classes=self.position_size).float() # (B, T, position_size)

        position_features = self.position_module(position_onehot)  # (B, T, hidden_size)

        action_features = self.translator(torch.cat([features, position_features], dim=-1))
        
        logits_flag = self.head_flag(action_features)    # (B, T, flag_size)
        logits_action = self.head_action(action_features) # (B, T, action_size)
        content_logits = self.head_content(torch.reshape(content_logits, (batch_size * context_size, self.channel, self.height, self.width)))

        content_logits = torch.reshape(content_logits, (batch_size, context_size, self.content_size))

        return logits_flag, logits_action, x_logits, y_logits, position_onehot, content_logits
    