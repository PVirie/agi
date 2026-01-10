import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np
import logging

from interfaces.network import Value_Network
from ..components.conv_resnet import ResNet, Bottleneck
from ..components.std_conv import ImpalaCNN
from utilities.safe_torch_module import Safe_nn_Module


class Value_Core(Value_Network, nn.Module, Safe_nn_Module):

    def __init__(self, action_size, position_size, width, height, channel, layers, device=None, persistence_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="value_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.flag_size = 5  # num classes for flag
        self.action_size = action_size
        self.position_size = position_size
        self.content_size = channel * width * height
        self.packed_action_size = 1 + 3 + self.content_size  # ext_flag + action + x + y + content

        self.width = width
        self.height = height
        self.channel = channel

        # self.conv_layers = ResNet(Bottleneck, layers, num_classes=1, num_channels=channel)
        self.conv_layers = ImpalaCNN(output_dims=1, input_channels=channel, width=width, height=height, depths=layers)

        self.reset_parameters()
        self.load()


    def reset_parameters(self):
        self.conv_layers.reset_parameters()


    def __compute(self, context):
        # context has shape (batch, context_size, 1 + position_size + content_size)
        # action has shape (batch, context_size, self.packed_action_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        # first slice the image content
        image_content = context[:, :, 1 + self.position_size:]
        image_part = torch.reshape(image_content, (batch_size * context_size, self.channel, self.height, self.width))

        values = self.conv_layers(image_part)  # (batch_size * context_size, 1)
        values = torch.reshape(values, (batch_size, context_size, 1))
        
        return values
    

    def get_latest_value(self, context):

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            value = self.__compute(context)
            
        return value[:, -1, ...].cpu().numpy()
    

    def get_value(self, context):

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        batch_size = context.size(0)
        context_size = context.size(1)
        value = self.__compute(context)

        # collapse last dimension
        batch_value = torch.reshape(value, (batch_size, context_size))

        return batch_value

