import torch
import torch.nn as nn
import numpy as np
import logging

from interfaces.network import Value_Network
from implementations.networks.torch.components.std_conv import ImpalaCNN
from utilities.safe_torch_module import Safe_nn_Module


class Value_Core(Value_Network, nn.Module, Safe_nn_Module):

    def __init__(self, 
                 position_size, 
                 width, height, channel, 
                 output_dims,
                 layers, 
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="conv_value_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.position_size = position_size
        self.content_size = channel * width * height

        self.width = width
        self.height = height
        self.channel = channel
        self.output_dims = output_dims

        self.conv_layers = ImpalaCNN(
            output_dims=32, 
            input_channels=channel, 
            width=width, 
            height=height, 
            depths=layers
        )
        
        self.read_out_layers = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.reset_parameters()
        self.load(override_persistence_path=first_load_path)
        self.eval()


    def reset_parameters(self):
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

        content = context[:, :, (1 + 1 + self.output_dims + self.position_size): ]  # (batch_size, context_size, content_size)
        
        image_part = torch.reshape(content, (batch_size * context_size, self.channel, self.height, self.width))
        image_features = self.conv_layers(image_part)  # (batch_size * context_size, conv_output_size)
        
        values = self.read_out_layers(image_features)  # (batch_size * context_size, 1)
        values = torch.reshape(values, (batch_size, context_size, 1))
        
        return values
    

    def get_value(self, context):

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        batch_size = context.size(0)
        context_size = context.size(1)
        value = self.compute(context)

        # collapse last dimension
        batch_value = torch.reshape(value, (batch_size, context_size))

        return batch_value

