import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np
import logging

from interfaces.network import Policy_Network
from implementations.networks.torch.components.base import init_weights, Categorical_With_Mask
from implementations.networks.torch.components.temporal_unet import TemporalUNet
from utilities.safe_torch_module import Safe_nn_Module


class Atari_Core(Policy_Network, nn.Module, Safe_nn_Module):

    def __init__(self, action_size, position_size, width, height, channel, hidden_size, layers, history_steps=0, max_temporal_len=32, device=None, persistence_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="atari_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.flag_size = 5  # num classes for flag
        self.action_size = action_size
        self.position_size = position_size
        self.content_size = channel * width * height
        self.packed_action_size = 1 + 1 + self.position_size + self.content_size  # ext_flag + action + position + content
        self.packed_context_size = 1 + self.position_size + self.content_size  # reward + position + content

        self.width = width
        self.height = height
        self.channel = channel
        self.hidden_size = hidden_size

        # feature always has size 32
        self.temporal_unet = TemporalUNet(
            n_channels=channel, vec_dim=1 + position_size, hidden_dim=hidden_size,
            bilinear=True, history_steps=history_steps, max_temporal_len=max_temporal_len)

        self.head_flag = nn.Sequential(
            nn.Linear(self.temporal_unet.out_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.flag_size)   # self.flag_size classes
        )
        self.head_action = nn.Sequential(
            nn.Linear(self.temporal_unet.out_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_size)   # action_size classes
        )
        self.head_content = nn.Sequential(
            nn.ReLU()
        )

        self.position_step = nn.Sequential(
            nn.Linear(position_size + action_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, position_size),
            nn.Sigmoid()
        )

        # add parameter for Normal distribution scale
        self.log_std = nn.Parameter(torch.zeros(1, 1, self.content_size))

        self.reset_parameters()
        self.load()
        self.eval()


    def reset_parameters(self):
        # Reset parameters of all layers
        self.temporal_unet.reset_parameters()

        self.head_flag.apply(init_weights)
        self.head_action.apply(init_weights)
        self.head_content.apply(init_weights)
        self.position_step.apply(init_weights)

        nn.init.constant_(self.log_std, 0.0)


    def __compute(self, context):
        # context has shape (batch, context_size, self.packed_context_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        # first slice the image content
        image_content = context[:, :, 1 + self.position_size: ]  # (batch_size, context_size, content_size)
        image_part = torch.reshape(image_content, (batch_size, context_size, self.channel, self.height, self.width))
        non_image_part = context[:, :, : 1 + self.position_size]  # (batch_size, context_size, 1 + position_size)
        last_position = context[:, :, 1: 1 + self.position_size]  # (batch_size, context_size, position_size)

        features, x_logits, y_logits, content_logits = self.temporal_unet(image_part, non_image_part)
        features = torch.reshape(features, (batch_size, context_size, self.temporal_unet.out_features))
        content_logits = torch.reshape(content_logits, (batch_size, context_size, self.content_size))
        
        return features, last_position, content_logits
    

    def get_action(self, context, valid_actions=None):

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, flag_size + action_size)
            available_flags = valid_actions[:, :, :self.flag_size].to(self.device)
            available_actions = valid_actions[:, :, self.flag_size:].to(self.device)

        batch_size = context.size(0)
        features, last_position, content_logits = self.__compute(context)

        logits_flag = self.head_flag(features)    # (B, T, flag_size)
        logits_action = self.head_action(features) # (B, T, action_size)
        pprobs_content = self.head_content(content_logits) # (B, T, content_size)

        props_flag = Categorical_With_Mask(logits=logits_flag, mask=available_flags)
        props_action = Categorical_With_Mask(logits=logits_action, mask=available_actions)
        props_content = Normal(loc=pprobs_content, scale=torch.exp(self.log_std))

        action_flag = props_flag.sample()
        action_action = props_action.sample()
        action_content = props_content.sample()

        # make one hot encoding for action, location
        action_onehot = torch.nn.functional.one_hot(action_action.long(), num_classes=self.action_size).float()
        logits_position = self.position_step(torch.concat([last_position, action_onehot], dim=-1))
        props_position = Bernoulli(probs=logits_position)
        next_position = props_position.sample()

        action = torch.cat([
            action_flag.unsqueeze(-1),
            action_action.unsqueeze(-1),
            next_position,
            action_content
        ], dim=-1)

        return action.cpu().numpy().astype(int)
    

    def get_log_probability(self, context, selected_action, valid_actions=None):

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, flag_size + action_size)
            available_flags = valid_actions[:, :, :self.flag_size].to(self.device)
            available_actions = valid_actions[:, :, self.flag_size:].to(self.device)

        batch_size = context.size(0)
        context_size = context.size(1)
        features, last_position, content_logits = self.__compute(context)

        logits_flag = self.head_flag(features)    # (B, T, flag_size)
        logits_action = self.head_action(features) # (B, T, action_size)
        pprobs_content = self.head_content(content_logits) # (B, T, content_size)

        props_flag = Categorical_With_Mask(logits=logits_flag, mask=available_flags)
        props_action = Categorical_With_Mask(logits=logits_action, mask=available_actions)
        props_content = Normal(loc=pprobs_content, scale=torch.exp(self.log_std))

        action_flag = selected_action[:, :, 0]
        action_action = selected_action[:, :, 1]
        current_position = selected_action[:, :, 2:2 + self.position_size]
        action_content = selected_action[:, :, 2 + self.position_size:]

        # make one hot encoding for action, location
        action_onehot = torch.nn.functional.one_hot(action_action.long(), num_classes=self.action_size).float()
        logits_position = self.position_step(torch.concat([last_position, action_onehot], dim=-1))
        props_position = Bernoulli(probs=logits_position)

        log_prob_flag = props_flag.log_prob(action_flag)
        log_prob_action = props_action.log_prob(action_action)
        log_prob_position = props_position.log_prob(current_position).mean(-1)
        log_prob_content = props_content.log_prob(action_content).mean(-1)
        
        entropy_flag = props_flag.entropy()
        entropy_action = props_action.entropy()
        entropy_position = props_position.entropy().mean(-1)
        entropy_content = props_content.entropy().mean(-1)

        return torch.stack([
            log_prob_flag, log_prob_action, log_prob_position, log_prob_content
        ], dim=-1), torch.stack([
            entropy_flag, entropy_action, entropy_position, entropy_content
        ], dim=-1)


    def unpack_action(self, packed_action):
        # packed_action has shape (batch, self.packed_action_size)
        # return int_action, ext_action, position, content

        int_part = packed_action[:, 0].astype(int)
        ext_part = packed_action[:, 1:2].astype(int)
        position = packed_action[:, 2:2 + self.position_size].astype(float)
        content = packed_action[:, 2 + self.position_size:].astype(float)

        return int_part, ext_part, position, content
    

    def pack_context(self, b_reward=None, b_position=None, b_content=None):
        # b_xxx has shape (batch, ...)
        # return packed_action_seq of shape (batch, self.packed_action_size) of type int
        # replace none with zeros

        batch_size = None
        if b_reward is not None:
            batch_size = b_reward.shape[0]
        elif b_position is not None:
            batch_size = b_position.shape[0]
        elif b_content is not None:
            batch_size = b_content.shape[0]
        else:
            raise ValueError("At least one of b_reward, b_content must be provided")
        
        if b_reward is None:
            b_reward = np.zeros((batch_size,), dtype=float)
        if b_position is None:
            b_position = np.zeros((batch_size, self.position_size), dtype=float)
        if b_content is None:
            b_content = np.zeros((batch_size, self.content_size), dtype=float)

        packed_context = np.concatenate([
            np.reshape(b_reward, (batch_size, 1)),
            b_position,
            b_content
        ], axis=-1).astype(float)

        return packed_context


# return only action log prob
class Action_Projector:
    def __init__(self, master_core):
        self.master_core = master_core

    def parameters(self):
        return self.master_core.parameters()
    
    def get_log_probability(self, context, selected_action, valid_actions=None):
        all_logprobs, all_entropy = self.master_core.get_log_probability(context, selected_action, valid_actions)
        log_probs = all_logprobs[:, :, [0, 1]].sum(dim=-1)  # sum over selected logprob components
        entropy = all_entropy[:, :, [0, 1]].sum(dim=-1)
        return log_probs, entropy
    
    def train(self):
        self.master_core.train()

    def eval(self):
        self.master_core.eval()
    
    
# return only content log prob
class Content_Projector:
    def __init__(self, master_core):
        self.master_core = master_core

    def parameters(self):
        return self.master_core.parameters()
    
    def get_log_probability(self, context, selected_action, valid_actions=None):
        all_logprobs, all_entropy = self.master_core.get_log_probability(context, selected_action, valid_actions)
        log_probs = all_logprobs[:, :, 3]  # content logprob
        entropy = all_entropy[:, :, 3]
        return log_probs, entropy
    
    def train(self):
        self.master_core.train()

    def eval(self):
        self.master_core.eval()
