import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np
import logging

from interfaces.network import Policy_Network
from ..components.base import init_weights, Categorical_With_Mask
from ..components.temporal_unet import TemporalUNet
from utilities.safe_torch_module import Safe_nn_Module


class Policy_Core(Policy_Network, nn.Module, Safe_nn_Module):

    def __init__(self, action_size, position_size, width, height, channel, hidden_size, layers, history_steps=0, max_temporal_len=32, device=None, persistence_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="policy_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.flag_size = 5  # num classes for flag
        self.action_size = action_size
        self.position_size = position_size
        self.content_size = channel * width * height
        self.packed_action_size = 1 + 3 + self.content_size  # ext_flag + action + x + y + content

        self.width = width
        self.height = height
        self.channel = channel
        self.hidden_size = hidden_size

        # feature always has size 32
        self.temporal_unet = TemporalUNet(
            n_channels=channel, vec_dim=1 + position_size, num_temporal_layers=layers, 
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
            nn.Sigmoid()
        )

        self.position_step = nn.Sequential(
            nn.Linear(position_size + action_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, position_size),
            nn.Sigmoid()
        )

        self.reset_parameters()
        self.load()


    def reset_parameters(self):
        # Reset parameters of all layers
        self.temporal_unet.reset_parameters()

        self.head_flag.apply(init_weights)
        self.head_action.apply(init_weights)
        self.head_content.apply(init_weights)
        self.position_step.apply(init_weights)


    def __compute(self, context, action):
        # context has shape (batch, context_size, 1 + position_size + content_size)
        # action has shape (batch, context_size, self.packed_action_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        # first slice the image content
        image_content = context[:, :, 1 + self.position_size:]
        image_part = torch.reshape(image_content, (batch_size, context_size, self.channel, self.height, self.width))
        non_image_part = context[:, :, :1 + self.position_size]

        features, x_logits, y_logits, content_logits = self.temporal_unet(image_part, non_image_part)
        features = torch.reshape(features, (batch_size, context_size, self.temporal_unet.out_features))
        x_logits = torch.reshape(x_logits, (batch_size, context_size, self.width))
        y_logits = torch.reshape(y_logits, (batch_size, context_size, self.height))
        content_logits = torch.reshape(content_logits, (batch_size, context_size, self.content_size))
        
        return features, x_logits, y_logits, content_logits
    

    def get_action(self, context, action, valid_actions=None):

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        if action is None:
            action = torch.zeros((context.size(0), context.size(1), self.packed_action_size), dtype=torch.float32).to(self.device)
        elif isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, flag_size + action_size)
            available_flags = valid_actions[:, :, :self.flag_size].to(self.device)
            available_actions = valid_actions[:, :, self.flag_size:].to(self.device)

        batch_size = context.size(0)
        features, x_logits, y_logits, content_logits = self.__compute(context, action)

        logits_flag = self.head_flag(features)    # (B, T, flag_size)
        logits_action = self.head_action(features) # (B, T, action_size)
        pprobs_content = self.head_content(content_logits) # (B, T, content_size)

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
            action_content
        ], dim=-1)

        # compute position
        last_position = context[:, :, 1:1 + self.position_size]
        # make one hot encoding for action, location
        action_onehot = torch.nn.functional.one_hot(action_action.long(), num_classes=self.action_size).float()
        logits_position = self.position_step(torch.concat([last_position, action_onehot], dim=-1))
        props_position = Bernoulli(probs=logits_position)
        position = props_position.sample()

        action = action.cpu().numpy().astype(int)
        position = position.cpu().numpy().astype(float)

        return action, position
    

    def get_log_probability(self, context, action, valid_actions=None, target_action=None, only_logprob_components=False):

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        if action is None:
            action = torch.zeros((context.size(0), context.size(1), self.packed_action_size), dtype=torch.float32).to(self.device)
        elif isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32).to(self.device)

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
        features, x_logits, y_logits, content_logits = self.__compute(context, action)

        logits_flag = self.head_flag(features)    # (B, T, flag_size)
        logits_action = self.head_action(features) # (B, T, action_size)
        pprobs_content = self.head_content(content_logits) # (B, T, content_size)

        props_flag = Categorical_With_Mask(logits=logits_flag, mask=available_flags)
        props_action = Categorical_With_Mask(logits=logits_action, mask=available_actions)
        props_x = Categorical(logits=x_logits)
        probs_y = Categorical(logits=y_logits)
        props_content = Bernoulli(probs=pprobs_content)

        if target_action is None:
            target_action = action

        action_flag = target_action[:, :, 0]
        action_action = target_action[:, :, 1]
        action_x = target_action[:, :, 2]
        action_y = target_action[:, :, 3]
        action_content = target_action[:, :, 4:]

        # compute position
        last_position = context[:, :, 1:1 + self.position_size]
        # make one hot encoding for action, location
        action_onehot = torch.nn.functional.one_hot(action_action.long(), num_classes=self.action_size).float()
        logits_position = self.position_step(torch.concat([last_position, action_onehot], dim=-1))
        props_position = Bernoulli(probs=logits_position)
        position = props_position.sample()

        log_prob_flag = props_flag.log_prob(action_flag)
        log_prob_action = props_action.log_prob(action_action)
        log_prob_x = props_x.log_prob(action_x)
        log_prob_y = probs_y.log_prob(action_y)
        log_prob_content = props_content.log_prob(action_content).mean(-1)
        log_prob_position = props_position.log_prob(position).mean(-1)
        batch_log_prob = log_prob_flag + log_prob_action + log_prob_x + log_prob_y + log_prob_content + log_prob_position
        
        entropy_flag = props_flag.entropy()
        entropy_action = props_action.entropy()
        entropy_x = props_x.entropy()
        entropy_y = probs_y.entropy()
        entropy_content = props_content.entropy().mean(-1)
        entropy_position = props_position.entropy().mean(-1)
        batch_entropy = entropy_flag + entropy_action + entropy_x + entropy_y + entropy_content + entropy_position


        if only_logprob_components:
            # collapse last dimension
            log_prob_flag = torch.reshape(log_prob_flag, (batch_size, context_size))
            log_prob_action = torch.reshape(log_prob_action, (batch_size, context_size))
            log_prob_x = torch.reshape(log_prob_x, (batch_size, context_size))
            log_prob_y = torch.reshape(log_prob_y, (batch_size, context_size))
            log_prob_content = torch.reshape(log_prob_content, (batch_size, context_size))
            log_prob_position = torch.reshape(log_prob_position, (batch_size, context_size))

            # return [log_prob_flag, log_prob_action, log_prob_x, log_prob_y, log_prob_content, log_prob_position]
            return torch.stack([
                log_prob_flag, log_prob_action, log_prob_x, log_prob_y, log_prob_content, log_prob_position
            ], dim=-1)
        else:
            # collapse last dimension
            batch_log_prob = torch.reshape(batch_log_prob, (batch_size, context_size))
            batch_entropy = torch.reshape(batch_entropy, (batch_size, context_size))

            return batch_log_prob, batch_entropy


    def unpack_action(self, packed_action):
        # packed_action has shape (batch, self.packed_action_size)
        # return int_action, ext_action , content

        int_part = packed_action[:, 0].astype(int)
        ext_part = packed_action[:, 1:4].astype(int)
        content = packed_action[:, 4:].astype(float)

        return int_part, ext_part, content
    

    def pack_action(self, b_int=None, b_ext=None, b_content=None):
        # b_xxx has shape (batch, ...)
        # return packed_action_seq of shape (batch, self.packed_action_size) of type int
        # replace none with zeros

        batch_size = None
        if b_int is not None:
            batch_size = b_int.shape[0]
        elif b_ext is not None:
            batch_size = b_ext.shape[0]
        elif b_content is not None:
            batch_size = b_content.shape[0]
        else:
            raise ValueError("At least one of b_int, b_ext, b_content must be provided")
        
        if b_int is None:
            b_int = np.zeros((batch_size,), dtype=int)
        if b_ext is None:
            b_ext = np.zeros((batch_size, 3), dtype=int)
        if b_content is None:
            b_content = np.zeros((batch_size, self.content_size), dtype=float)

        packed_action = np.concatenate([
            np.reshape(b_int, (batch_size, 1)),
            b_ext,
            b_content
        ], axis=-1).astype(int)

        return packed_action
    
