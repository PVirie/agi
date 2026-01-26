import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np
import logging

from interfaces.network import Policy_Network
from implementations.networks.torch.components.base import init_weights
from implementations.networks.torch.components.base import Categorical_With_Mask
from implementations.networks.torch.components.std_conv import ImpalaCNN
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
        self.packed_action_size = 1 + 3 + position_size + self.content_size  # int_flag + action + x + y + position + content
        self.packed_context_size = 1 + 1 + 3 + position_size + self.content_size  # reward + packed_action_size

        self.width = width
        self.height = height
        self.channel = channel
        self.hidden_size = hidden_size

        self.stem = ImpalaCNN(
            output_dims=hidden_size,
            input_channels=channel,
            width=width,
            height=height,
            depths=[16, 32, 64]
        )

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

        self.reset_parameters()
        self.load()
        self.eval()


    def reset_parameters(self):
        # Reset parameters of all layers
        self.stem.reset_parameters()

        def init_actor_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.head_flag.apply(init_actor_weights)
        self.head_action.apply(init_actor_weights)


    def __compute(self, context):
        # context has shape (batch, context_size, self.packed_context_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        # first slice the image content
        image_content = context[:, :, (1 + 1 + 3 + self.position_size): ]  # (batch_size, context_size, content_size)
        image_part = torch.reshape(image_content, (batch_size, context_size, self.channel, self.height, self.width))
        reward_action_part = context[:, :, :(1 + 1 + 3)]  # (batch_size, context_size, 1 + 1 + 3)
        last_position = context[:, :, (1 + 1 + 3): (1 + 1 + 3 + self.position_size)]  # (batch_size, context_size, position_size)

        # make one hot encoding for action, location
        reward = reward_action_part[:, :, 0:1]  # (batch_size, context_size, 1)
        flag_onehot = torch.nn.functional.one_hot(reward_action_part[:, :, 1].long(), num_classes=self.flag_size).float()
        action_onehot = torch.nn.functional.one_hot(reward_action_part[:, :, 2].long(), num_classes=self.action_size).float()
        x_onehot = torch.nn.functional.one_hot(reward_action_part[:, :, 3].long(), num_classes=self.width).float()
        y_onehot = torch.nn.functional.one_hot(reward_action_part[:, :, 4].long(), num_classes=self.height).float()
        
        image_part = torch.reshape(image_part, (batch_size * context_size, self.channel, self.height, self.width))
        features = self.stem(image_part)
        features = torch.reshape(features, (batch_size, context_size, -1))  # (B, T, hidden_size)
        
        logits_flag = self.head_flag(features)    # (B, T, flag_size)
        logits_action = self.head_action(features) # (B, T, action_size)

        x_logits = torch.zeros((batch_size, context_size, self.width), device=self.device)
        y_logits = torch.zeros((batch_size, context_size, self.height), device=self.device)
        next_position = last_position.clone()
        pprobs_content = torch.sigmoid(torch.zeros((batch_size, context_size, self.content_size), device=self.device))

        return logits_flag, logits_action, x_logits, y_logits, next_position, pprobs_content
    

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

        logits_flag, logits_action, x_logits, y_logits, next_position, pprobs_content = self.__compute(context)

        props_flag = Categorical_With_Mask(logits=logits_flag, mask=available_flags)
        props_action = Categorical_With_Mask(logits=logits_action, mask=available_actions)
        props_x = Categorical(logits=x_logits)
        probs_y = Categorical(logits=y_logits)
        props_content = Bernoulli(probs=pprobs_content)

        action_flag = props_flag.sample()
        action_action = props_action.sample()
        action_x = props_x.sample()
        action_y = probs_y.sample()
        action_content = props_content.sample()

        action = torch.cat([
            action_flag.unsqueeze(-1),
            action_action.unsqueeze(-1),
            action_x.unsqueeze(-1),
            action_y.unsqueeze(-1),
            next_position.detach(),
            action_content
        ], dim=-1)

        return action.cpu().numpy().astype(float)
    

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

        logits_flag, logits_action, x_logits, y_logits, next_position, pprobs_content = self.__compute(context)

        props_flag = Categorical_With_Mask(logits=logits_flag, mask=available_flags)
        props_action = Categorical_With_Mask(logits=logits_action, mask=available_actions)
        props_x = Categorical(logits=x_logits)
        probs_y = Categorical(logits=y_logits)
        props_content = Bernoulli(probs=pprobs_content)

        action_flag = selected_action[:, :, 0]
        action_action = selected_action[:, :, 1]
        action_x = selected_action[:, :, 2]
        action_y = selected_action[:, :, 3]
        action_content = selected_action[:, :, (1 + 3 + self.position_size):]

        log_prob_flag = props_flag.log_prob(action_flag)
        log_prob_action = props_action.log_prob(action_action)
        log_prob_x = props_x.log_prob(action_x)
        log_prob_y = probs_y.log_prob(action_y)
        log_prob_content = props_content.log_prob(action_content).mean(-1)
        
        entropy_flag = props_flag.entropy()
        entropy_action = props_action.entropy()
        entropy_x = props_x.entropy()
        entropy_y = probs_y.entropy()
        entropy_content = props_content.entropy().mean(-1)

        return torch.stack([
            log_prob_flag, log_prob_action, log_prob_x, log_prob_y, log_prob_content
        ], dim=-1), torch.stack([
            entropy_flag, entropy_action, entropy_x, entropy_y, entropy_content
        ], dim=-1)


    def unpack_action(self, packed_action):
        # packed_action has shape (batch, self.packed_action_size)
        # return int_action, ext_action, position, content

        int_part = packed_action[:, 0].astype(int)
        ext_part = packed_action[:, 1:4].astype(int)
        position = packed_action[:, 4:4 + self.position_size].astype(float)
        content = packed_action[:, 4 + self.position_size:].astype(float)

        return int_part, ext_part, position, content
    

    def pack_context(self, b_reward=None, b_int=None, b_ext=None, b_position=None, b_content=None):
        # b_xxx has shape (batch, ...)
        # return packed_action_seq of shape (batch, self.packed_action_size) of type int
        # replace none with zeros

        batch_size = None
        if b_reward is not None:
            batch_size = b_reward.shape[0]
        elif b_int is not None:
            batch_size = b_int.shape[0]
        elif b_ext is not None:
            batch_size = b_ext.shape[0]
        elif b_position is not None:
            batch_size = b_position.shape[0]
        elif b_content is not None:
            batch_size = b_content.shape[0]
        else:
            raise ValueError("At least one of b_reward, b_content must be provided")
        
        if b_reward is None:
            b_reward = np.zeros((batch_size,), dtype=float)
        if b_int is None:
            b_int = np.zeros((batch_size,), dtype=int)
        if b_ext is None:
            b_ext = np.zeros((batch_size, 3), dtype=int)
        if b_position is None:
            b_position = np.zeros((batch_size, self.position_size), dtype=float)
        if b_content is None:
            b_content = np.zeros((batch_size, self.content_size), dtype=float)

        packed_context = np.concatenate([
            np.reshape(b_reward, (batch_size, 1)),
            np.reshape(b_int, (batch_size, 1)),
            b_ext,
            b_position,
            b_content
        ], axis=-1).astype(float)

        return packed_context


# return only action log prob
class Projector:
    def __init__(self, master_core, selected_indices=[0, 1, 2, 3]):
        self.master_core = master_core
        self.selected_indices = selected_indices

    def parameters(self):
        return self.master_core.parameters()
    
    def get_log_probability(self, context, selected_action, valid_actions=None):
        all_logprobs, all_entropy = self.master_core.get_log_probability(context, selected_action, valid_actions)
        log_probs = all_logprobs[:, :, self.selected_indices].sum(dim=-1)  # sum over selected logprob components
        entropy = all_entropy[:, :, self.selected_indices].sum(dim=-1)
        return log_probs, entropy
    
    def train(self):
        self.master_core.train()

    def eval(self):
        self.master_core.eval()
