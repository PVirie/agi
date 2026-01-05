import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np
import logging

from interfaces.core import Core
from .base import init_weights, Categorical_With_Mask
from .sfstct import SpatialEncoder, TemporalEncoder
from .temporal_unet import TemporalUNet
from utilities.safe_torch_module import Safe_nn_Module


class Action_Content_Core(Core, nn.Module, Safe_nn_Module):

    def __init__(self, action_size, position_size, width, height, channel, hidden_size, layers, max_temporal_range=32, device=None, persistence_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="core", device=device, persistence_path=persistence_path)
        self.device = device

        self.flag_size = 5  # num classes for flag
        self.action_size = action_size
        self.position_size = position_size
        self.content_size = channel * width * height
        self.packed_action_size = 1 + 2 + self.content_size  # ext_flag + action + loc + content

        self.width = width
        self.height = height
        self.channel = channel
        self.hidden_size = hidden_size

        # feature always has size 32
        self.temporal_unet = TemporalUNet(
            n_channels=channel, vec_dim=1 + position_size, num_temporal_layers=layers, 
            bilinear=True, max_temporal_len=32)

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
        self.head_value = nn.Sequential(
            nn.Linear(self.temporal_unet.out_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)   # Regression output
        )

        self.value_logstd = nn.Parameter(torch.zeros(1, 1))

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
        self.head_value.apply(init_weights)

        self.position_step.apply(init_weights)

        # make sure value_logstd is initialized to 0
        nn.init.constant_(self.value_logstd, 0.0)


    def __compute(self, context, action):
        # context has shape (batch, context_size, 1 + position_size + content_size)
        # action has shape (batch, context_size, self.packed_action_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        # first slice the image content
        image_content = context[:, :, 1 + self.position_size:]
        image_part = torch.reshape(image_content, (batch_size, context_size, self.channel, self.height, self.width))
        non_image_part = context[:, :, :1 + self.position_size]

        features, heatmap_logits, content_logits = self.temporal_unet(image_part, non_image_part)
        heatmap_logits = torch.reshape(heatmap_logits, (batch_size, context_size, self.height * self.width))
        content_logits = torch.reshape(content_logits, (batch_size, context_size, self.content_size))
        
        return features, heatmap_logits, content_logits
    

    def get_latest_value(self, context, action):
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.int64).to(self.device)
        elif action is None:
            action = torch.zeros((context.size(0), context.size(1), self.packed_action_size), dtype=torch.int64).to(self.device)

        with torch.no_grad():
            features, _, _ = self.__compute(context, action)
            logits_value = self.head_value(features)    # (B, T, 1)
            
        return logits_value[:, -1, ...].cpu().numpy()
    

    def get_action_and_value(self, context, action, valid_actions=None, use_action=False, use_grad=True):

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
        features, heatmap_logits, content_logits = self.__compute(context, action)

        logits_flag = self.head_flag(features)    # (B, T, flag_size)
        logits_action = self.head_action(features) # (B, T, action_size)
        pprobs_content = self.head_content(content_logits) # (B, T, content_size)
        value = self.head_value(features)    # (B, T, 1)

        props_flag = Categorical_With_Mask(logits=logits_flag, mask=available_flags)
        props_action = Categorical_With_Mask(logits=logits_action, mask=available_actions)
        props_loc = Categorical(logits=heatmap_logits)
        props_content = Bernoulli(probs=pprobs_content)

        if use_action:
            action_flag = action[:, :, 0]
            action_action = action[:, :, 1]
            action_loc = action[:, :, 2]
            action_content = action[:, :, 3:]
        else:
            action_flag = props_flag.sample()
            action_action = props_action.sample()
            action_loc = props_loc.sample()
            action_content = props_content.sample()

            action = torch.cat([
                action_flag.unsqueeze(-1),
                action_action.unsqueeze(-1),
                action_loc.unsqueeze(-1),
                action_content
            ], dim=-1)


        # compute position
        last_position = context[:, :, 1:1 + self.position_size]
        # make one hot encoding for action, location
        action_onehot = torch.nn.functional.one_hot(action_action.long(), num_classes=self.action_size).float()
        logits_position = self.position_step(torch.concat([last_position, action_onehot], dim=-1))
        props_position = Bernoulli(probs=logits_position)
        position = props_position.sample()


        log_prob_flag = props_flag.log_prob(action_flag)
        log_prob_action = props_action.log_prob(action_action)
        log_prob_loc = props_loc.log_prob(action_loc)
        log_prob_content = props_content.log_prob(action_content).mean(-1)
        log_prob_position = props_position.log_prob(position).mean(-1)
        batch_log_prob = log_prob_flag + log_prob_action + log_prob_loc + log_prob_content + log_prob_position
        
        entropy_flag = props_flag.entropy()
        entropy_action = props_action.entropy()
        entropy_loc = props_loc.entropy()
        entropy_content = props_content.entropy().mean(-1)
        entropy_position = props_position.entropy().mean(-1)
        batch_entropy = entropy_flag + entropy_action + entropy_loc + entropy_content + entropy_position

        # collapse last dimension
        batch_log_prob = torch.reshape(batch_log_prob, (batch_size, -1))
        batch_entropy = torch.reshape(batch_entropy, (batch_size, -1))
        batch_value = torch.reshape(value, (batch_size, -1))

        action = action.cpu().numpy().astype(int)
        position = position.cpu().numpy().astype(float)

        if not use_grad:
            batch_log_prob = batch_log_prob.detach().cpu().numpy()
            batch_entropy = batch_entropy.detach().cpu().numpy()
            batch_value = batch_value.detach().cpu().numpy()

        return action, position, batch_log_prob, batch_entropy, batch_value


    def get_log_probability(self, context, action, valid_actions=None, target_action=None, f_mask=None):
        """
        context has shape (batch, context_size, ...)
        action has shape (batch, context_size, self.packed_action_size)
        valid_actions has shape (batch, context_size, flag_size + action_size)
        target_action has shape (batch, context_size, self.packed_action_size)
        f_mask has shape (batch, context_size, 4)
        """

        batch_size = context.size(0)
        context_size = context.size(1)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        if action is None:
            action = torch.zeros((batch_size, context_size, self.packed_action_size), dtype=torch.float32).to(self.device)
        elif isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32).to(self.device)

        if f_mask is None:
            f_mask = torch.ones((batch_size, context_size, 4), dtype=torch.float32).to(self.device)
        elif isinstance(f_mask, np.ndarray):
            f_mask = torch.tensor(f_mask, dtype=torch.float32).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, flag_size + action_size)
            available_flags = valid_actions[:, :, :self.flag_size].to(self.device)
            available_actions = valid_actions[:, :, self.flag_size:].to(self.device)

        features, heatmap_logits, content_logits = self.__compute(context, action)

        logits_flag = self.head_flag(features)    # (B, T, flag_size)
        logits_action = self.head_action(features) # (B, T, action_size)
        pprobs_content = self.head_content(content_logits) # (B, T, content_size)

        props_flag = Categorical_With_Mask(logits=logits_flag, mask=available_flags)
        props_action = Categorical_With_Mask(logits=logits_action, mask=available_actions)
        props_loc = Categorical(logits=heatmap_logits)
        props_content = Bernoulli(probs=pprobs_content)

        if target_action is None:
            target_action = action

        action_flag = target_action[:, :, 0]
        action_action = target_action[:, :, 1]
        action_loc = target_action[:, :, 2]
        action_content = target_action[:, :, 3:]

        log_prob_flag = props_flag.log_prob(action_flag)
        log_prob_action = props_action.log_prob(action_action)
        log_prob_loc = props_loc.log_prob(action_loc)
        log_prob_content = props_content.log_prob(action_content).mean(-1)

        log_prob = torch.stack([log_prob_flag, log_prob_action, log_prob_loc, log_prob_content], dim=-1)
        masked_log_prob = log_prob * f_mask
        sum_log_prob = torch.sum(masked_log_prob, dim=-1)

        return sum_log_prob


    def unpack_action(self, packed_action):
        # packed_action has shape (batch, self.packed_action_size)
        # return int_action, ext_action , content

        int_part = packed_action[:, 0].astype(int)
        pre_ext_part = packed_action[:, 1:3].astype(int)
        content = packed_action[:, 3:].astype(float)

        # split pre_ext_part int of height times width into flag, x, y
        ext_part = np.zeros((packed_action.shape[0], 3), dtype=int)
        ext_part[:, 0] = pre_ext_part[:, 0]
        ext_part[:, 1] = pre_ext_part[:, 1] % self.width
        ext_part[:, 2] = pre_ext_part[:, 1] // self.width

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

        b_ext = np.reshape(b_ext, (batch_size, 3))
        b_pack_ext = np.zeros((batch_size, 2), dtype=int)
        b_pack_ext[:, 0] = b_ext[:, 0]
        b_pack_ext[:, 1] = b_ext[:, 1] + b_ext[:, 2] * self.width

        packed_action = np.concatenate([
            np.reshape(b_int, (batch_size, 1)),
            b_pack_ext,
            b_content
        ], axis=-1).astype(int)

        return packed_action
    
