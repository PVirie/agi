import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np
import logging

from implementations.networks.torch.components.base import init_weights
from implementations.networks.torch.components.base import Categorical_With_Mask
from implementations.networks.torch.components.std_conv import ImpalaCNN
from implementations.networks.torch.policy.base import Policy_Core as Base_Policy_Core, Projector as Base_Projector
from utilities.safe_torch_module import Safe_nn_Module


class Policy_Core(Base_Policy_Core):

    def __init__(self, 
                 mem_ops_size, action_size, position_size, 
                 width, height, channel, 
                 hidden_size, layers, 
                 history_steps=0, max_temporal_len=32, 
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="dualism_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.flag_size = mem_ops_size  # num classes for flag
        self.action_size = action_size
        self.position_size = position_size
        self.content_size = channel * width * height
        self.packed_action_size = 1 + 1 + position_size + self.content_size  # int_flag + action + x + y + position + content
        self.packed_context_size = 1 + 1 + 1 + position_size + self.content_size  # reward + packed_action_size

        self.width = width
        self.height = height
        self.channel = channel
        self.hidden_size = hidden_size

        self.conv_layers = ImpalaCNN(output_dims=hidden_size, input_channels=channel, width=width, height=height, depths=layers)
        vec_dim = self.flag_size + action_size + position_size

        self.position_step = nn.Sequential(
            nn.Linear(vec_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, position_size),
        )

        self.obs_feature_to_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, position_size)
        )

        self.obs_feature_to_logvar = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, position_size)
        )

        self.head_flag = nn.Sequential(
            nn.Linear(position_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.flag_size)   # self.flag_size classes
        )
        self.head_action = nn.Sequential(
            nn.Linear(position_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_size)   # action_size classes
        )

        self.reset_parameters()
        self.load(override_persistence_path=first_load_path)
        self.eval()


    def slow_parameters(self):
        # override to return all parameters except the recognize_module parameters
        params = []
        for name, param in self.named_parameters():
            if "recognize_module" not in name:
                params.append(param)
        return params


    def fast_parameters(self):
        # override to return only the recognize_module parameters
        params = []
        for name, param in self.named_parameters():
            if "recognize_module" in name:
                params.append(param)
        return params


    def reset_parameters(self):
        # Reset parameters of all layers
        self.conv_layers.reset_parameters()

        def init_actor_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.position_step.apply(init_actor_weights)
        self.obs_feature_to_mean.apply(init_actor_weights) # need near zero initialization for stable training with reparameterization trick
        self.obs_feature_to_logvar.apply(init_actor_weights) # need near zero initialization for stable training with reparameterization trick
        self.head_flag.apply(init_actor_weights)
        self.head_action.apply(init_actor_weights)


    def compute(self, context):
        # context has shape (batch, context_size, self.packed_context_size)
        batch_size = context.size(0)
        context_size = context.size(1)

        # first slice the image content
        image_content = context[:, :, (1 + 1 + 1 + self.position_size): ]  # (batch_size, context_size, content_size)
        image_part = torch.reshape(image_content, (batch_size, context_size, self.channel, self.height, self.width))
        last_position = context[:, :, (1 + 1 + 1): (1 + 1 + 1 + self.position_size)]  # (batch_size, context_size, position_size)

        # make one hot encoding for action, location
        reward = context[:, :, 0:1]  # (batch_size, context_size, 1)
        flag_onehot = torch.nn.functional.one_hot(context[:, :, 1].long(), num_classes=self.flag_size).float()
        action_onehot = torch.nn.functional.one_hot(context[:, :, 2].long(), num_classes=self.action_size).float()
        
        vec = torch.concat([flag_onehot, action_onehot, last_position], dim=-1)  # (batch_size, context_size, flag_size + action_size + position_size)
        step_position_logits = self.position_step(vec) + last_position  # (batch_size, context_size, position_size)

        image_part = torch.reshape(image_part, (batch_size * context_size, self.channel, self.height, self.width))
        obs_logits = self.conv_layers(image_part)  # (batch_size * context_size, conv_output_size)
        obs_logits = torch.reshape(obs_logits, (batch_size, context_size, self.hidden_size))  # (batch_size, context_size, conv_output_size)

        # use reparameterize trick to sample position
        obs_means = self.obs_feature_to_mean(obs_logits)  # (batch_size, context_size, position_size)
        obs_logvars = self.obs_feature_to_logvar(obs_logits)  # (batch_size, context_size, position_size)
        std = torch.exp(0.5 * obs_logvars) # convert log-var to std
        epsilon = torch.randn_like(std) # sample random noise
        obs_position_logits = obs_means + epsilon * std

        vae_loss = torch.sum(
            torch.exp(obs_logvars) + obs_means**2 - 1.0 - obs_logvars,
            dim=-1
        )  # (B, T)

        positions = step_position_logits + obs_position_logits # (batch_size, context_size, position_size)

        logits_flag = self.head_flag(positions)    # (B, T, flag_size)
        logits_action = self.head_action(positions) # (B, T, action_size)

        # # check nan
        # if torch.isnan(logits_flag).any() or torch.isnan(logits_action).any() or torch.isnan(positions).any() or torch.isnan(vae_loss).any():
        #     logging.warning(f"logits_flag: {logits_flag}")
        #     logging.warning(f"logits_action: {logits_action}")
        #     logging.warning(f"positions: {positions}")
        #     logging.warning(f"vae_loss: {vae_loss}")
        #     logging.warning("NaN detected in compute()")

        return logits_flag, logits_action, positions, vae_loss
    

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

        logits_flag, logits_action, positions, _ = self.compute(context)

        probs_flag = Categorical_With_Mask(logits=logits_flag, mask=available_flags)
        probs_action = Categorical_With_Mask(logits=logits_action, mask=available_actions)

        batch_size = context.size(0)
        context_size = context.size(1)

        action_flag = probs_flag.sample()
        action_action = probs_action.sample()
        action_content = np.zeros((batch_size, context_size, self.content_size), dtype=np.float32)
        action = np.concatenate([
            action_flag.unsqueeze(-1).cpu().numpy(),
            action_action.unsqueeze(-1).cpu().numpy(),
            positions.detach().cpu().numpy(),
            action_content
        ], axis=-1)

        return action.astype(float)
    

    def get_log_probability(self, context, selected_action, valid_actions=None):
        # context has shape (batch, context_size, self.packed_context_size)

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

        logits_flag, logits_action, positions, _ = self.compute(context)

        probs_flag = Categorical_With_Mask(logits=logits_flag, mask=available_flags)
        probs_action = Categorical_With_Mask(logits=logits_action, mask=available_actions)

        action_flag = selected_action[:, :, 0]
        action_action = selected_action[:, :, 1]

        log_prob_flag = probs_flag.log_prob(action_flag)
        log_prob_action = probs_action.log_prob(action_action)
        
        entropy_flag = probs_flag.entropy()
        entropy_action = probs_action.entropy()

        return torch.stack([
            log_prob_flag, log_prob_action
        ], dim=-1), torch.stack([
            entropy_flag, entropy_action
        ], dim=-1)


    def get_log_probability_with_aux_loss(self, context, selected_action, valid_actions=None):
        # now context has shape (batch, context_size + 1, self.packed_context_size)

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)

        context_full = context
        context = context_full[:, :-1, :]  # remove last context for computing logprob
        # target_position = context_full[:, 1:, (1 + 1 + 1):(1 + 1 + 1 + self.position_size)]  # only position part for position loss

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, flag_size + action_size)
            available_flags = valid_actions[:, :, :self.flag_size].to(self.device)
            available_actions = valid_actions[:, :, self.flag_size:].to(self.device)

        logits_flag, logits_action, positions, aux_loss = self.compute(context)

        probs_flag = Categorical_With_Mask(logits=logits_flag, mask=available_flags)
        probs_action = Categorical_With_Mask(logits=logits_action, mask=available_actions)

        action_flag = selected_action[:, :, 0]
        action_action = selected_action[:, :, 1]

        log_prob_flag = probs_flag.log_prob(action_flag)
        log_prob_action = probs_action.log_prob(action_action)
        
        entropy_flag = probs_flag.entropy()
        entropy_action = probs_action.entropy()

        return torch.stack([
            log_prob_flag, log_prob_action
        ], dim=-1), torch.stack([
            entropy_flag, entropy_action
        ], dim=-1), aux_loss


    def unpack_action(self, packed_action):
        # packed_action has shape (batch, self.packed_action_size)
        # return int_action, ext_action, position, content

        int_part = packed_action[:, 0].astype(int)
        ext_part = packed_action[:, 1:2].astype(int)
        position = packed_action[:, 2:2 + self.position_size].astype(float)
        content = packed_action[:, 2 + self.position_size:].astype(float)

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
            b_ext = np.zeros((batch_size, 1), dtype=int)
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


# return only selected statistics
class Projector(Base_Projector):

    def slow_parameters(self):
        return self.master_core.slow_parameters()
    
    def fast_parameters(self):
        return self.master_core.fast_parameters()