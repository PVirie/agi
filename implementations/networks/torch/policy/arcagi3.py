import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np
import logging

from interfaces.network import Policy_Network
from implementations.networks.torch.components.base import init_weights
from implementations.networks.torch.components.base import Categorical_With_Mask
from implementations.networks.torch.components.temporal_unet import TemporalUNet
from implementations.networks.torch.policy.base import Policy_Core as Base_Policy_Core
from utilities.safe_torch_module import Safe_nn_Module


class Policy_Core(Base_Policy_Core):

    def __init__(self, 
                 int_action_size, ext_action_size, position_size, 
                 width, height, channel, 
                 hidden_size, layers, 
                 history_steps=0, max_temporal_len=32, 
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="arcagi3_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.int_action_size = int_action_size  # num classes for flag
        self.ext_action_size = ext_action_size
        self.position_size = position_size
        self.content_size = channel * width * height
        self.packed_action_size = 1 + 3 + position_size + self.content_size  # int_action_size + ext_action_size + x + y + position + content
        self.packed_context_size = 1 + 1 + 3 + position_size + self.content_size  # reward + packed_action_size

        self.width = width
        self.height = height
        self.channel = channel
        self.hidden_size = hidden_size

        vec_dim = 1 + self.int_action_size + ext_action_size + position_size
        self.temporal_unet = TemporalUNet(
            output_dims=hidden_size,
            input_channels=channel, width=width, height=height,
            vec_dim=vec_dim, hidden_dim=hidden_size,
            depths=layers, history_steps=history_steps, max_temporal_len=max_temporal_len)

        self.head_flag = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.int_action_size)   # int_action_size classes
        )
        self.head_action = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, ext_action_size)   # ext_action_size classes
        )
        self.position_step = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, position_size)
        )
        self.head_content = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1)
        )

        self.content_logstd = nn.Parameter(torch.zeros(1, 1, self.content_size))

        self.reset_parameters()
        self.load(override_persistence_path=first_load_path)
        self.eval()


    def reset_parameters(self):
        # Reset parameters of all layers
        self.temporal_unet.reset_parameters()

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
        self.position_step.apply(init_actor_weights)
        self.head_content.apply(init_actor_weights)
        nn.init.constant_(self.content_logstd, 0.0)


    def compute(self, context):
        # context has shape (batch, context_size, self.packed_context_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        reward = context[:, :, 0:1]  # (batch_size, context_size, 1)
        flag_onehot = torch.nn.functional.one_hot(context[:, :, 1].long(), num_classes=self.int_action_size).float()
        action_onehot = torch.nn.functional.one_hot(context[:, :, 2].long(), num_classes=self.ext_action_size).float()
        x_onehot = torch.nn.functional.one_hot(context[:, :, 3].long(), num_classes=self.width).float()
        y_onehot = torch.nn.functional.one_hot(context[:, :, 4].long(), num_classes=self.height).float()
        last_position = context[:, :, (1 + 1 + 3): (1 + 1 + 3 + self.position_size)]  # (batch_size, context_size, position_size)
        content = context[:, :, (1 + 1 + 3 + self.position_size): ]  # (batch_size, context_size, content_size)

        vec = torch.concat([reward, flag_onehot, action_onehot, last_position], dim=-1)  # (batch_size, context_size, 1 + int_action_size + ext_action_size + position_size)
        image_part = torch.reshape(content, (batch_size, context_size, self.channel, self.height, self.width))
        features, x_logits, y_logits, content_logits = self.temporal_unet(image_part, vec)
        
        logits_flag = self.head_flag(features)    # (B, T, int_action_size)
        logits_action = self.head_action(features) # (B, T, ext_action_size)
        position_logits = self.position_step(features)
        content_logits = self.head_content(torch.reshape(content_logits, (batch_size * context_size, self.channel, self.height, self.width)))

        content_logits = torch.reshape(content_logits, (batch_size, context_size, self.content_size))

        return logits_flag, logits_action, x_logits, y_logits, position_logits, content_logits
    

    def get_action(self, context, valid_actions=None):

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
            available_flags = valid_actions[:, :, :self.int_action_size].to(self.device)
            available_actions = valid_actions[:, :, self.int_action_size:].to(self.device)

        logits_flag, logits_action, x_logits, y_logits, position_logits, content_logits = self.compute(context)

        probs_flag = Categorical_With_Mask(logits=logits_flag, mask=available_flags)
        probs_action = Categorical_With_Mask(logits=logits_action, mask=available_actions)
        probs_x = Categorical(logits=x_logits)
        probs_y = Categorical(logits=y_logits)
        probs_position = Bernoulli(logits=position_logits)
        probs_content = Normal(content_logits, torch.exp(self.content_logstd.expand_as(content_logits)))

        action_flag = probs_flag.sample()
        action_action = probs_action.sample()
        action_x = probs_x.sample()
        action_y = probs_y.sample()
        next_position = probs_position.sample()
        action_content = probs_content.sample()

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
        # context has shape (batch, context_size, self.packed_context_size)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
            available_flags = valid_actions[:, :, :self.int_action_size].to(self.device)
            available_actions = valid_actions[:, :, self.int_action_size:].to(self.device)

        logits_flag, logits_action, x_logits, y_logits, position_logits, content_logits = self.compute(context)

        probs_flag = Categorical_With_Mask(logits=logits_flag, mask=available_flags)
        probs_action = Categorical_With_Mask(logits=logits_action, mask=available_actions)
        probs_x = Categorical(logits=x_logits)
        probs_y = Categorical(logits=y_logits)
        probs_position = Bernoulli(logits=position_logits)
        probs_content = Normal(content_logits, torch.exp(self.content_logstd.expand_as(content_logits)))

        action_flag = selected_action[:, :, 0]
        action_action = selected_action[:, :, 1]
        action_x = selected_action[:, :, 2]
        action_y = selected_action[:, :, 3]
        action_position = selected_action[:, :, (1 + 3):(1 + 3 + self.position_size)]
        action_content = selected_action[:, :, (1 + 3 + self.position_size):]

        log_prob_flag = probs_flag.log_prob(action_flag)
        log_prob_action = probs_action.log_prob(action_action)
        log_prob_x = probs_x.log_prob(action_x)
        log_prob_y = probs_y.log_prob(action_y)
        log_prob_position = probs_position.log_prob(action_position).mean(-1)
        log_prob_content = probs_content.log_prob(action_content).mean(-1)
        
        entropy_flag = probs_flag.entropy()
        entropy_action = probs_action.entropy()
        entropy_x = probs_x.entropy()
        entropy_y = probs_y.entropy()
        entropy_position = probs_position.entropy().mean(-1)
        entropy_content = probs_content.entropy().mean(-1)

        return torch.stack([
            log_prob_flag, log_prob_action, log_prob_x, log_prob_y, log_prob_position, log_prob_content
        ], dim=-1), torch.stack([
            entropy_flag, entropy_action, entropy_x, entropy_y, entropy_position, entropy_content
        ], dim=-1)


    def get_log_probability_with_aux_loss(self, context, selected_action, valid_actions=None):
        # now context has shape (batch, context_size + 1, self.packed_context_size)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        context_full = context
        context = context_full[:, :-1, :]  # remove last context for computing logprob
        # target_position = context_full[:, 1:, (1 + 1 + 3):(1 + 1 + 3 + self.position_size)]  # only position part for position loss
        target_content = context_full[:, 1:, (1 + 1 + 3 + self.position_size): ]  # only content part for action loss

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
            available_flags = valid_actions[:, :, :self.int_action_size].to(self.device)
            available_actions = valid_actions[:, :, self.int_action_size:].to(self.device)

        logits_flag, logits_action, x_logits, y_logits, position_logits, content_logits = self.compute(context)

        probs_flag = Categorical_With_Mask(logits=logits_flag, mask=available_flags)
        probs_action = Categorical_With_Mask(logits=logits_action, mask=available_actions)
        probs_x = Categorical(logits=x_logits)
        probs_y = Categorical(logits=y_logits)
        probs_position = Bernoulli(logits=position_logits)
        probs_content = Normal(content_logits, torch.exp(self.content_logstd.expand_as(content_logits)))

        action_flag = selected_action[:, :, 0]
        action_action = selected_action[:, :, 1]
        action_x = selected_action[:, :, 2]
        action_y = selected_action[:, :, 3]
        action_position = selected_action[:, :, (1 + 3):(1 + 3 + self.position_size)]
        action_content = selected_action[:, :, (1 + 3 + self.position_size):]

        log_prob_flag = probs_flag.log_prob(action_flag)
        log_prob_action = probs_action.log_prob(action_action)
        log_prob_x = probs_x.log_prob(action_x)
        log_prob_y = probs_y.log_prob(action_y)
        log_prob_position = probs_position.log_prob(action_position).mean(-1)
        log_prob_content = probs_content.log_prob(action_content).mean(-1)
        
        entropy_flag = probs_flag.entropy()
        entropy_action = probs_action.entropy()
        entropy_x = probs_x.entropy()
        entropy_y = probs_y.entropy()
        entropy_position = probs_position.entropy().mean(-1)
        entropy_content = probs_content.entropy().mean(-1)

        # svl_unsum_loss
        svl_unsum_loss = torch.mean((probs_content.mean - target_content) ** 2, dim=-1)

        return torch.stack([
            log_prob_flag, log_prob_action, log_prob_x, log_prob_y, log_prob_position, log_prob_content
        ], dim=-1), torch.stack([
            entropy_flag, entropy_action, entropy_x, entropy_y, entropy_position, entropy_content
        ], dim=-1), svl_unsum_loss


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
