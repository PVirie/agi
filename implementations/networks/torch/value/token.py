import torch
import torch.nn as nn
import numpy as np
import logging

from mambapy.mamba import Mamba as Mamba, MambaConfig as MambaConfig

from implementations.networks.torch.components.base import init_weights
from interfaces.network import Value_Network
from implementations.networks.torch.components.std_resnet import ResNet
from utilities.safe_torch_module import Safe_nn_Module


class Value_Core(Value_Network, nn.Module, Safe_nn_Module):

    def __init__(self, 
                 position_size, content_size,
                 output_dims,
                 dict_size, embedding_dim,
                 hidden_size, layers, 
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="token_value_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.position_size = position_size
        self.content_size = content_size
        self.packed_context_size = 1 + 1 + output_dims + position_size + self.content_size  # reward + packed_action_size

        self.embedding = nn.Embedding(dict_size, embedding_dim)  # for direction token
        
        # self.backbone = ResNet(output_dims=hidden_size, input_dims=embedding_dim * self.packed_context_size, hidden_dims=hidden_size, layers=layers)
        self.adapter = nn.Linear(embedding_dim * self.packed_context_size, hidden_size)
        config = MambaConfig(d_model=hidden_size, n_layers=layers)
        self.backbone = Mamba(config)

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
        self.adapter.reset_parameters()
        #self.backbone.reset_parameters()
        self.backbone.apply(init_weights)

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

        embedded = self.embedding(context.long())  # (batch_size, context_size, packed_context_size, embedding_dim)
        embedded = embedded.view(batch_size, context_size, -1)  # (batch_size, context_size, content_size * embedding_dim)
        
        #features = self.backbone(embedded)  # (batch_size, context_size, hidden_size)
        vec = self.adapter(embedded)  # (batch_size, context_size, hidden_size)
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

