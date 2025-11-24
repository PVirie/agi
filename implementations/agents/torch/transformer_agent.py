import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from interfaces.agent import Agent_Core
from core.torch.base import Multilayer_Relu, apply_transformer, causal_mask, reset_transformer_decoder


class Transformer_Agent(Agent_Core, nn.Module):

    def __init__(self, action_space, hidden_size, heads, layers, device):
        super().__init__()

        self.device = device
        self.hidden_size = hidden_size

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=heads, device=device)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)

        self.critic = Multilayer_Relu(hidden_size, 1, hidden_size, 2, device=device)
        self.actor_mean = Multilayer_Relu(hidden_size, action_space, hidden_size, 2, device=device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_space))

        self.reset_parameters()


    def reset_parameters(self):
        # Reset parameters of all layers
        reset_transformer_decoder(self.decoder)
        self.critic.reset_parameters()


    def get_value(self, x):
        context_size = x.size(1)
        x = apply_transformer(self.decoder, torch.reshape(x, (-1, context_size, self.hidden_size)))

        return self.critic(x)


    def get_action_and_value(self, x, action=None):
        context_size = x.size(1)
        x = apply_transformer(self.decoder, torch.reshape(x, (-1, context_size, self.hidden_size)))

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
