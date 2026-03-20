import torch
import torch.nn as nn
import numpy as np
import logging

from implementations.networks.torch.components.base import init_weights
from interfaces.network import Value_Network
from implementations.networks.torch.components.std_resnet import ResNet
from utilities.safe_torch_module import Safe_nn_Module


class Value_Core(Value_Network, nn.Module, Safe_nn_Module):

    def __init__(self, 
                 int_action_size, ext_action_size,
                 position_size, content_size,
                 output_dims,
                 dict_size, embedding_dim,
                 hidden_size, layers, 
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="token_value_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.int_action_size = int_action_size  # num classes for flag
        self.action_size = ext_action_size
        self.position_size = position_size
        self.content_size = content_size
        self.packed_action_size = 1 + output_dims + position_size + self.content_size  # int_flag + action + ... + position + content
        self.packed_context_size = 1 + 1 + output_dims + position_size + self.content_size  # reward + packed_action_size

        self.hidden_size = hidden_size
        self.output_dims = output_dims

        self.embedding = nn.Embedding(dict_size, embedding_dim)  # for direction token
        vec_dim = 1 + ext_action_size + self.position_size + content_size * embedding_dim
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
        content = context[:, :, (1 + 1 + self.output_dims + self.position_size): ]  # (batch_size, context_size, content_size)
        
        embedded = self.embedding(content.long())  # (batch_size, context_size, packed_context_size, embedding_dim)
        embedded = embedded.view(batch_size, context_size, -1)  # (batch_size, context_size, content_size * embedding_dim)
        
        vec = torch.concat([reward, action_onehot, last_position, embedded], dim=-1)  # (batch_size, context_size, 1 + ext_action_size + position_size + content_size * embedding_dim)
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

