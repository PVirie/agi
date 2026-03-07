import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from implementations.networks.torch.components.base import init_weights
from implementations.networks.torch.components.std_conv import ImpalaCNN
from utilities.safe_torch_module import Safe_nn_Module
from implementations.networks.torch.value.conv import Value_Core as Base_Value_Core


class Value_Core(Base_Value_Core):

    def __init__(self, 
                 int_action_size, ext_action_size, position_size, 
                 width, height, channel, 
                 hidden_size, layers, 
                 history_steps=0, max_temporal_len=32, 
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="mix_aggregate_value_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.int_action_size = int_action_size  # num classes for flag
        self.action_size = ext_action_size
        self.position_size = position_size
        self.content_size = channel * width * height
        self.packed_action_size = 1 + 3 + position_size + self.content_size  # int_flag + action + x + y + position + content
        self.packed_context_size = 1 + 1 + 3 + position_size + self.content_size  # reward + packed_action_size

        self.width = width
        self.height = height
        self.channel = channel
        self.hidden_size = hidden_size
        self.history_steps = history_steps

        vec_dim = 1 + self.int_action_size + self.position_size  # reward + flag_onehot + position
        self.vec_embedding = nn.Sequential(
            nn.Linear(vec_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.conv_layers = ImpalaCNN(output_dims=hidden_size, input_channels=channel, width=width, height=height, depths=layers)

        self.read_out_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, 1)
        )

        self.aggregator = torch.ones((1, 1, history_steps + 1), device=device)  # for temporal aggregation of values

        self.reset_parameters()
        self.load(override_persistence_path=first_load_path)
        self.eval()


    def reset_parameters(self):
        self.vec_embedding.apply(init_weights)
        self.conv_layers.reset_parameters()

        def init_value_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.read_out_layers.apply(init_value_weights)


    def compute(self, context):
        # context has shape (batch, context_size, packed_context_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        # first slice the image content
        image_content = context[:, :, (1 + 1 + 3 + self.position_size): ]  # (batch_size, context_size, content_size)
        image_part = torch.reshape(image_content, (batch_size * context_size, self.channel, self.height, self.width))
        last_position = context[:, :, (1 + 1 + 3): (1 + 1 + 3 + self.position_size)]  # (batch_size, context_size, position_size)

        # make one hot encoding for action, location
        reward = context[:, :, 0:1]  # (batch_size, context_size, 1)
        flag_onehot = torch.nn.functional.one_hot(context[:, :, 1].long(), num_classes=self.int_action_size).float()
        action_onehot = torch.nn.functional.one_hot(context[:, :, 2].long(), num_classes=self.action_size).float()
        x_onehot = torch.nn.functional.one_hot(context[:, :, 3].long(), num_classes=self.width).float()
        y_onehot = torch.nn.functional.one_hot(context[:, :, 4].long(), num_classes=self.height).float()

        vec = torch.concat([reward, flag_onehot, last_position], dim=-1)  # (batch_size, context_size, 1 + int_action_size + position_size)
        embedded_features = self.vec_embedding(vec)  # (batch_size, context_size, hidden_size)

        image_features = self.conv_layers(image_part)  # (batch_size * context_size, hidden_size)
        image_features = torch.reshape(image_features, (batch_size, context_size, self.hidden_size))  # (batch_size, context_size, hidden_size)

        features = torch.concat([embedded_features, image_features], dim=-1)  # (batch_size, context_size, hidden_size * 2)

        values = self.read_out_layers(features)  # (batch_size, context_size, 1)
        values = torch.reshape(values, (batch_size, context_size, 1))

        # 1. Reshape to [batch, channels, timestep] for Conv1d
        values_conv = values.transpose(1, 2)

        # 2. Manual left padding: (padding_left, padding_right)
        # We pad (n - 1) zeros to the left to keep the output length equal to 'timestep'
        values_padded = F.pad(values_conv, (self.history_steps, 0))

        # 3. Apply Conv1d with the aggregator kernel
        values_output = F.conv1d(values_padded, self.aggregator, bias=None, stride=1)

        # 4. Reshape back to [batch, timestep, 1]
        values_output = values_output.transpose(1, 2)
        
        return values_output