import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from interfaces.core import Core
from .base import Multilayer_Relu, Multilayer_CNN, apply_transformer, causal_mask, reset_transformer_decoder


class Transformer_Core(Core, nn.Module):

    def __init__(self, action_space, hidden_size, heads, layers, device):
        super().__init__()

        self.device = device
        self.hidden_size = hidden_size

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=heads, device=device)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)

        self.critic = Multilayer_Relu(hidden_size, 1, hidden_size, 2, device=device)
        self.actor_mean = Multilayer_Relu(hidden_size, action_space, hidden_size, 2, device=device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_space))

        self.projector = Multilayer_CNN(3, hidden_size, hidden_size, 2, kernel_size=3, device=device)

        self.reset_parameters()


    def reset_parameters(self):
        # Reset parameters of all layers
        reset_transformer_decoder(self.decoder)
        self.critic.reset_parameters()
        self.actor_mean.reset_parameters()
        self.projector.reset_parameters()


    def __compute(self, x):
        # x has shape (batch, context_size, channels, height, width)
        batch_size = x.size(0)
        context_size = x.size(1)
        x = self.projector(torch.reshape(x, (-1, x.size(2), x.size(3), x.size(4))))  # (batch * context_size, hidden_size)
        # x now has shape (batch * context_size, hidden_size)
        x = apply_transformer(self.decoder, torch.reshape(x, (batch_size, context_size, self.hidden_size)))
        # x now has shape (batch, context_size, hidden_size)
        return x
    

    def get_latest_value(self, x):
        return self.critic(self.__compute(x))[:, -1]
    

    def get_action_and_value(self, x, action=None):
        x = self.__compute(x)

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
