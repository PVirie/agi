import torch
import torch.nn as nn

from implementations.networks.torch.components.base import init_weights

# non-convolutional ResNet components, reflect impala structure but without conv layers, to be used in non-image-based models
class ResBlock(nn.Module):
    
    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = channels
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Linear(channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, channels)
        )

    def forward(self, x):
        return x + self.block(x)
    

class ResNet(nn.Module):

    def __init__(self, output_dims, input_dims, hidden_dims, layers):
        super().__init__()
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.layers = layers

        self.input_layer = nn.Linear(input_dims, hidden_dims)
        self.res_blocks = nn.Sequential(*[ResBlock(hidden_dims, layer_dim) for layer_dim in layers])
        self.output_layer = nn.Linear(hidden_dims, output_dims)


    def reset_parameters(self):
        self.input_layer.apply(init_weights)
        for block in self.res_blocks:
            block.block.apply(init_weights)
        self.output_layer.apply(init_weights)

        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x
    
