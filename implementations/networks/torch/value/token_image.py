import torch
import torch.nn as nn
import numpy as np
import logging

from implementations.networks.torch.components.base import init_weights
from interfaces.network import Value_Network
from implementations.networks.torch.components.std_resnet import ResNet
from implementations.networks.torch.components.std_conv import ImpalaCNN, ImpalaCNN1D
from utilities.safe_torch_module import Safe_nn_Module


class Value_Core(Value_Network, nn.Module, Safe_nn_Module):

    def __init__(self, 
                 int_action_size, ext_action_size,
                 position_size,
                 output_dims,
                 token_part_size,
                 dict_size, embedding_dim,
                 width, height, channel,
                 hidden_size, layers, 
                 history_steps=0, max_temporal_len=32, 
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="token_image_value_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.token_part_size = token_part_size
        self.embedding_dim = embedding_dim

        self.width = width
        self.height = height
        self.channel = channel
        image_part_size = width * height * channel

        self.int_action_size = int_action_size  # num classes for flag
        self.action_size = ext_action_size
        self.position_size = position_size
        self.content_size = token_part_size + image_part_size
        self.packed_action_size = 1 + output_dims + position_size + self.content_size  # int_flag + action + ... + position + content
        self.packed_context_size = 1 + 1 + output_dims + position_size + self.content_size  # reward + packed_action_size

        self.hidden_size = hidden_size
        self.output_dims = output_dims

        self.embedding = nn.Embedding(dict_size, embedding_dim)  # for tokens
        self.image_embedding = nn.Embedding(256, 4)  # for image pixels, shared across channels
        self.feature_channel = self.channel * 4
        self.conv_layers = ImpalaCNN(
            output_dims=hidden_size, 
            input_channels=self.feature_channel, 
            width=width, height=height,
            depths=[64, 64, 128]
        )

        self.conv1d_layers = ImpalaCNN1D(
            output_dims=hidden_size,
            input_channels=embedding_dim,
            seq_length=token_part_size,
            depths=[16, 32, 32]
        )

        vec_dim = hidden_size + hidden_size
        self.backbone = ResNet(output_dims=hidden_size, input_dims=vec_dim, hidden_dims=hidden_size, layers=layers)

        self.read_out_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.reset_parameters()
        self.load(override_persistence_path=first_load_path)
        self.eval()


    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.image_embedding.reset_parameters()
        self.conv_layers.reset_parameters()
        self.conv1d_layers.reset_parameters()
        self.backbone.reset_parameters()

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

        reward = context[:, :, 0:1]  # (batch_size, context_size, 1)
        flag_onehot = torch.nn.functional.one_hot(context[:, :, 1].long(), num_classes=self.int_action_size).float()
        action_onehot = torch.nn.functional.one_hot(context[:, :, 2].long(), num_classes=self.action_size).float()
        last_position = context[:, :, (1 + 1 + self.output_dims): (1 + 1 + self.output_dims + self.position_size)]  # (batch_size, context_size, position_size)
        tokens = context[:, :, (1 + 1 + 1 + self.position_size): (1 + 1 + 1 + self.position_size + self.token_part_size)]  # (batch_size, context_size, token_part_size)
        image_content = context[:, :, (1 + 1 + 1 + self.position_size + self.token_part_size): ]  # (batch_size, context_size, image_part_size)

        embedded = self.embedding(tokens.long())  # (batch_size, context_size, packed_context_size, embedding_dim)
        token_features = torch.reshape(embedded, (batch_size * context_size, self.token_part_size, self.embedding_dim))  # (batch_size * context_size, token_part_size, embedding_dim)
        token_features = token_features.permute(0, 2, 1)  # (batch_size * context_size, embedding_dim, token_part_size)
        token_features = self.conv1d_layers(token_features)  # (batch_size * context_size, hidden_size)
        token_features = token_features.view(batch_size, context_size, self.hidden_size)  # (batch_size, context_size, hidden_size)

        # process image content through conv layers
        image_embedded = self.image_embedding(image_content.long())  # (batch_size, context_size, image_part_size, embedding_dim)
        obs_features = torch.reshape(image_embedded, (batch_size * context_size, self.height, self.width, self.feature_channel))  # (batch_size * context_size, height, width, channel * embedding_dim)
        obs_features = obs_features.permute(0, 3, 1, 2)  # (batch_size * context_size, channel * embedding_dim, height, width)
        obs_features = self.conv_layers(obs_features)  # (batch_size * context_size, hidden_size)
        obs_features = obs_features.view(batch_size, context_size, self.hidden_size)  # (batch_size, context_size, hidden_size)
        
        vec = torch.concat([token_features, obs_features], dim=-1)  # (batch_size, context_size, hidden_size + hidden_size)
        features = self.backbone(vec)  # (batch_size, context_size, hidden_size)
        
        values = self.read_out_layers(features)  # (batch_size, context_size, 1)
        
        return values
    

    def get_value(self, context):

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.int64).to(self.device)

        batch_size = context.size(0)
        context_size = context.size(1)
        value = self.compute(context)

        # collapse last dimension
        batch_value = torch.reshape(value, (batch_size, context_size))

        return batch_value

