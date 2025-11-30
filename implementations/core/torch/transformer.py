import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

from interfaces.core import Core
from .base import Multilayer_Relu, Multilayer_CNN, apply_transformer, causal_mask, reset_transformer_decoder


class Transformer_Core(Core, nn.Module):

    def __init__(self, action_size, position_size, width, height, channel, hidden_size, heads, layers, device):
        super().__init__()
        # content_size = channels x height x width

        self.action_size = action_size
        self.position_size = position_size
        self.content_size = channel * width * height

        self.width = width
        self.height = height
        self.channel = channel

        self.hidden_size = hidden_size
        self.device = device

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=heads, device=device)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)

        self.critic = Multilayer_Relu(hidden_size, 1, hidden_size, 2, device=device)
        self.actor_mean = Multilayer_Relu(hidden_size, 1 + action_size + self.content_size, hidden_size, 2, device=device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, 1, 1 + action_size + self.content_size))

        self.image_projector = Multilayer_CNN(channel, hidden_size, hidden_size, 2, kernel_size=3, device=device)
        self.projector = Multilayer_Relu(1 + position_size + hidden_size, hidden_size, hidden_size, 2, device=device)

        self.position_step = Multilayer_Relu(position_size + action_size, position_size, hidden_size, 2, device=device)

        self.reset_parameters()


    def reset_parameters(self):
        # Reset parameters of all layers
        reset_transformer_decoder(self.decoder)
        self.critic.reset_parameters()
        self.actor_mean.reset_parameters()
        self.image_projector.reset_parameters()
        self.projector.reset_parameters()
        self.position_step.reset_parameters()

        # make sure actor_logstd is initialized to 0
        nn.init.zeros_(self.actor_logstd)


    def __compute(self, x):
        # x has shape (batch, context_size, 1 + position_size + content_size)
        batch_size = x.size(0)
        context_size = x.size(1)

        # first slice the image content
        image_content = x[:, :, 1 + self.position_size:]
        image_content = torch.reshape(image_content, (batch_size, context_size, self.channel, self.height, self.width))
        image_part = self.image_projector(torch.reshape(image_content, (-1, self.channel, self.height, self.width)))
        
        non_image_part = x[:, :, :1 + self.position_size]

        x = torch.concat([non_image_part, torch.reshape(image_part, (batch_size, context_size, -1))], dim=2)
        x = self.projector(torch.reshape(x, (-1, x.size(2))))
        x = torch.reshape(x, (batch_size, context_size, self.hidden_size))
        x = apply_transformer(self.decoder, x)
        # x now has shape (batch, context_size, hidden_size)
        return x
    

    def get_latest_value(self, x):
        return self.critic(self.__compute(x))[:, -1]
    

    def get_action_and_value(self, x, action=None):
        x = self.__compute(x)
        batch_size = x.size(0)
        action_size = 1 + self.action_size + self.content_size
        x = torch.reshape(x, (-1, self.hidden_size))
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        
        batch_action = torch.reshape(action, (batch_size, -1, action_size))
        batch_log_prob = torch.reshape(probs.log_prob(action).sum(1), (batch_size, -1))
        batch_entropy = torch.reshape(probs.entropy().sum(1), (batch_size, -1))
        batch_value = torch.reshape(self.critic(x), (batch_size, -1))

        return batch_action, batch_log_prob, batch_entropy, batch_value


    def unpack_action(self, packed_action, x):
        # packed_action has shape (batch, 1 + action_size + content_size)
        # x has shape (batch, context_size, 1 + position_size + content_size)
        # return ext_flag, action_data, position, content

        ext_part = packed_action[:, 0]
        action_data = packed_action[:, 1:1 + self.action_size]

        last_position = x[:, -1, 1:1 + self.position_size]
        position = self.position_step(torch.concat([last_position, action_data], dim=1))

        content = packed_action[:, 1 + self.action_size:]

        return ext_part, action_data, position, content