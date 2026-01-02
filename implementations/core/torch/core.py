import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np
import logging

from interfaces.core import Core
from .base import init_weights, Categorical_With_Mask_Sample
from .sfstct import SpatialEncoder, TemporalEncoder
from .temporal_unet import TemporalUNet
from utilities.safe_torch_module import Safe_nn_Module


class Action_Content_Core(Core, nn.Module, Safe_nn_Module):

    def __init__(self, action_size, position_size, width, height, channel, hidden_size, layers, device=None, persistence_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="core", device=device, persistence_path=persistence_path)
        self.device = device

        self.action_size = action_size
        self.position_size = position_size
        self.content_size = channel * width * height
        self.packed_action_size = 1 + 3 + self.content_size  # ext_flag + action + x + y + content

        self.width = width
        self.height = height
        self.channel = channel
        self.hidden_size = hidden_size

        # feature always has size 32
        self.temporal_unet = TemporalUNet(n_channels=channel, vec_dim=1 + position_size, num_temporal_layers=layers, bilinear=True)

        self.head_flag = nn.Sequential(
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 5)   # 5 classes
        )
        self.head_action = nn.Sequential(
            nn.Linear(32, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_size)   # action_size classes
        )
        self.head_content = nn.Sequential(
            nn.Sigmoid()
        )
        self.head_value = nn.Sequential(
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 1)   # Regression output
        )

        self.value_logstd = nn.Parameter(torch.zeros(1, 1))

        # self.position_step = Multilayer_Relu(position_size + action_size + width + height, position_size, hidden_size, 2)
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

        x_p, y_p, features, content_logits = self.temporal_unet(image_part, non_image_part)
        content_logits = torch.reshape(content_logits, (batch_size, context_size, self.content_size))
        
        return x_p, y_p, features, content_logits
    

    def get_latest_value(self, context, action):
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.int64).to(self.device)
        elif action is None:
            action = torch.zeros((context.size(0), context.size(1), self.packed_action_size), dtype=torch.int64).to(self.device)

        with torch.no_grad():
            _, _, features, _ = self.__compute(context, action)
            logits_value = self.head_value(features)    # (B, T, 1)
            
        return logits_value[:, -1, ...].cpu().numpy()
    

    def get_action_and_value(self, context, action, use_action=False, use_grad=True, extra_params=None):

        available_actions = None
        if extra_params is not None:
            available_actions = extra_params.get("available_actions", None)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        if action is None:
            action = torch.zeros((context.size(0), context.size(1), self.packed_action_size), dtype=torch.float32).to(self.device)
        elif isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32).to(self.device)

        batch_size = context.size(0)
        x_p, y_p, features, content_logits = self.__compute(context, action)

        logits_flag = self.head_flag(features)    # (B, T, 5)
        logits_action = self.head_action(features) # (B, T, 6)
        pprobs_content = self.head_content(content_logits) # (B, T, content_size)
        value = self.head_value(features)    # (B, T, 1)

        props_flag = Categorical(logits=logits_flag)
        props_action = Categorical_With_Mask_Sample(logits=logits_action)
        props_x = Categorical(probs=x_p)
        props_y = Categorical(probs=y_p)
        props_content = Bernoulli(probs=pprobs_content)

        if use_action:
            action_flag = action[:, :, 0]
            action_action = action[:, :, 1]
            action_x = action[:, :, 2]
            action_y = action[:, :, 3]
            action_content = action[:, :, 4:]
        else:
            action_flag = props_flag.sample()
            action_action = props_action.sample_from_available_indices(indices=available_actions)
            action_x = props_x.sample()
            action_y = props_y.sample()
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
        # make one hot encoding for action, x, y
        action_onehot = torch.nn.functional.one_hot(action_action.long(), num_classes=self.action_size).float()
        # x_onehot = torch.nn.functional.one_hot(action_x.long(), num_classes=self.width).float()
        # y_onehot = torch.nn.functional.one_hot(action_y.long(), num_classes=self.height).float()
        # position = self.position_step(torch.concat([last_position, action_onehot, x_onehot, y_onehot], dim=-1))
        logits_position = self.position_step(torch.concat([last_position, action_onehot], dim=-1))
        props_position = Bernoulli(probs=logits_position)
        position = props_position.sample()


        log_prob_flag = props_flag.log_prob(action_flag)
        log_prob_action = props_action.log_prob(action_action)
        log_prob_x = props_x.log_prob(action_x)
        log_prob_y = props_y.log_prob(action_y)
        log_prob_content = props_content.log_prob(action_content).mean(-1)
        log_prob_position = props_position.log_prob(position).mean(-1)
        batch_log_prob = log_prob_flag + log_prob_action + log_prob_x + log_prob_y + log_prob_content + log_prob_position
        

        entropy_flag = props_flag.entropy()
        entropy_action = props_action.entropy()
        entropy_x = props_x.entropy()
        entropy_y = props_y.entropy()
        entropy_content = props_content.entropy().mean(-1)
        entropy_position = props_position.entropy().mean(-1)
        batch_entropy = entropy_flag + entropy_action + entropy_x + entropy_y + entropy_content + entropy_position

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


    def get_log_probability(self, context, action, target_action=None, f_mask=None):
        """
        context has shape (batch, context_size, ...)
        action has shape (batch, context_size, self.packed_action_size)
        target_action has shape (batch, context_size, self.packed_action_size)
        f_mask has shape (batch, context_size, 5)
        """

        batch_size = context.size(0)
        context_size = context.size(1)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        if action is None:
            action = torch.zeros((batch_size, context_size, 1 + 3 + self.content_size), dtype=torch.float32).to(self.device)
        elif isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32).to(self.device)

        if f_mask is None:
            f_mask = torch.ones((batch_size, context_size, 5), dtype=torch.float32).to(self.device)
        elif isinstance(f_mask, np.ndarray):
            f_mask = torch.tensor(f_mask, dtype=torch.float32).to(self.device)

        x_p, y_p, features, content_logits = self.__compute(context, action)

        logits_flag = self.head_flag(features)    # (B, T, 5)
        logits_action = self.head_action(features) # (B, T, 6)
        pprobs_content = self.head_content(content_logits) # (B, T, content_size)

        props_flag = Categorical(logits=logits_flag)
        props_action = Categorical(logits=logits_action)
        props_x = Categorical(probs=x_p)
        props_y = Categorical(probs=y_p)
        props_content = Bernoulli(probs=pprobs_content)

        if target_action is None:
            target_action = action

        action_flag = target_action[:, :, 0]
        action_action = target_action[:, :, 1]
        action_x = target_action[:, :, 2]
        action_y = target_action[:, :, 3]
        action_content = target_action[:, :, 4:]

        log_prob_flag = props_flag.log_prob(action_flag)
        log_prob_action = props_action.log_prob(action_action)
        log_prob_x = props_x.log_prob(action_x)
        log_prob_y = props_y.log_prob(action_y)
        log_prob_content = props_content.log_prob(action_content).mean(-1)

        log_prob = torch.stack([log_prob_flag, log_prob_action, log_prob_x, log_prob_y, log_prob_content], dim=-1)
        masked_log_prob = log_prob * f_mask
        sum_log_prob = torch.sum(masked_log_prob, dim=-1)

        return sum_log_prob


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
            b_int.reshape((batch_size, 1)),
            b_ext.reshape((batch_size, 3)),
            b_content
        ], axis=-1).astype(int)

        return packed_action
    
