import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli
import numpy as np
import logging

from interfaces.network import Policy_Index_Network
from implementations.networks.torch.components.base import init_weights
from implementations.networks.torch.components.base import Categorical_With_Mask
from implementations.networks.torch.components.std_conv import ImpalaCNN
from utilities.safe_torch_module import Safe_nn_Module


class Policy_Core(Policy_Index_Network, nn.Module, Safe_nn_Module):

    def __init__(self, 
                 int_action_size, ext_action_size, position_size, 
                 working_memory_size,
                 width, height, channel, 
                 hidden_size, layers, 
                 device=None, 
                 persistence_path=None, first_load_path=None):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="working_memory_core", device=device, persistence_path=persistence_path)
        self.device = device

        self.int_action_size = int_action_size
        self.ext_action_size = ext_action_size
        self.position_size = position_size
        self.content_size = channel * width * height
        self.packed_action_size = 1 + 1 + position_size + self.content_size  # int_act + ext_act + position + content
        self.packed_context_size = 1 + 1 + 1 + position_size + self.content_size  # reward + int_act + ext_act + position + content

        self.working_memory_size = working_memory_size
        self.width = width
        self.height = height
        self.channel = channel
        self.hidden_size = hidden_size

        self.conv_layers = ImpalaCNN(output_dims=hidden_size, input_channels=channel, width=width, height=height, depths=layers)

        vec_dim = 1 + int_action_size + ext_action_size + position_size + hidden_size

        self.backbone = nn.Sequential(
            nn.Linear(vec_dim * working_memory_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )

        self.head_int = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, int_action_size)   # int_action_size classes
        )
        self.head_ext = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, ext_action_size)   # ext_action_size classes
        )
        self.head_position = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, position_size),
            nn.Sigmoid()
        )

        self.reset_parameters()
        self.load(override_persistence_path=first_load_path)
        self.eval()


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

        self.head_int.apply(init_actor_weights)
        self.head_ext.apply(init_actor_weights)
        self.head_position.apply(init_actor_weights)


    def compute(self, context, indices):
        # context has shape (batch, episodic_memory_size, self.packed_context_size)
        # indices has shape (batch, context_size, working_memory_size) with value in [0, episodic_memory_size - 1]
        batch_size = context.size(0)
        episodic_memory_size = context.size(1)
        context_size = indices.size(1)

        # first slice the image content
        image_content = context[:, :, (1 + 1 + 1 + self.position_size): ]  # (batch_size, episodic_memory_size, content_size)
        image_part = torch.reshape(image_content, (batch_size * episodic_memory_size, self.channel, self.height, self.width))
        obs_logits = self.conv_layers(image_part)  # (batch_size * episodic_memory_size, conv_output_size)
        obs_logits = torch.reshape(obs_logits, (batch_size, episodic_memory_size, self.hidden_size))  # (batch_size, episodic_memory_size, conv_output_size)

        # make one hot encoding for action, location
        reward = context[:, :, 0:1]  # (batch_size, episodic_memory_size, 1)
        int_onehot = torch.nn.functional.one_hot(context[:, :, 1].long(), num_classes=self.int_action_size).float()
        ext_onehot = torch.nn.functional.one_hot(context[:, :, 2].long(), num_classes=self.ext_action_size).float()
        last_position = context[:, :, (1 + 1 + 1): (1 + 1 + 1 + self.position_size)]  # (batch_size, episodic_memory_size, position_size)
        
        vec = torch.concat([reward, int_onehot, ext_onehot, last_position, obs_logits], dim=-1)  # (batch_size, episodic_memory_size, vec_dim)

        # vec has shape (batch_size, episodic_memory_size, vec_dim)
        # indices has shape (batch_size, context_size, working_memory_size) with value in [0, episodic_memory_size - 1]
        # we want to gather vec according to indices to get shape (batch_size, context_size, working_memory_size, vec_dim), 
        # then reshape to (batch_size, context_size, working_memory_size * vec_dim) for feeding into backbone
        batch_idx = torch.arange(batch_size, device=self.device).view(batch_size, 1, 1)
        gathered_vec = vec[batch_idx, indices] # (batch_size, context_size, working_memory_size, vec_dim)
        gathered_vec = torch.reshape(gathered_vec, (batch_size, context_size, self.working_memory_size * vec.size(-1)))  # (batch_size, context_size, working_memory_size * vec_dim)

        backbone_output = self.backbone(gathered_vec)  # (batch_size, context_size, hidden_size)

        int_logits = self.head_int(backbone_output)  # (batch_size, context_size, int_action_size)
        ext_logits = self.head_ext(backbone_output)  # (batch_size, context_size, ext_action_size)
        position_logits = self.head_position(backbone_output)  # (batch_size, context_size, position_size)

        return int_logits, ext_logits, position_logits
    

    def get_action(self, context, indices, valid_actions=None):

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)
        if isinstance(indices, np.ndarray):
            indices = torch.tensor(indices, dtype=torch.long).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
            available_flags = valid_actions[:, :, :self.int_action_size].to(self.device)
            available_actions = valid_actions[:, :, self.int_action_size:].to(self.device)

        logits_int, logits_ext, position_logits = self.compute(context, indices)

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)
        probs_position = Bernoulli(probs=position_logits)

        batch_size = context.size(0)
        context_size = indices.size(1)

        action_int = probs_int.sample()
        action_ext = probs_ext.sample()
        next_position = probs_position.sample()
        action_content = np.zeros((batch_size, context_size, self.content_size), dtype=np.float32)

        action = np.concatenate([
            action_int.unsqueeze(-1).cpu().numpy(),
            action_ext.unsqueeze(-1).cpu().numpy(),
            next_position.cpu().numpy(),
            action_content
        ], axis=-1)

        return action.astype(float)
    

    def get_log_probability(self, context, indices, selected_action, valid_actions=None):
        # context has shape (batch, episodic_memory_size, self.packed_context_size)
        # indices has shape (batch, context_size, working_memory_size) with value in [0, episodic_memory_size - 1]

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)
        if isinstance(indices, np.ndarray):
            indices = torch.tensor(indices, dtype=torch.long).to(self.device)

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
            available_flags = valid_actions[:, :, :self.int_action_size].to(self.device)
            available_actions = valid_actions[:, :, self.int_action_size:].to(self.device)

        logits_int, logits_ext, position_logits = self.compute(context, indices)

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)
        probs_position = Bernoulli(probs=position_logits)

        action_int = selected_action[:, :, 0]
        action_ext = selected_action[:, :, 1]
        action_position = selected_action[:, :, (1 + 1): (1 + 1 + self.position_size)]

        log_prob_int = probs_int.log_prob(action_int)
        log_prob_ext = probs_ext.log_prob(action_ext)
        log_prob_position = probs_position.log_prob(action_position).mean(-1)
        
        entropy_int = probs_int.entropy()
        entropy_ext = probs_ext.entropy()
        entropy_position = probs_position.entropy().mean(-1)

        return torch.stack([
            log_prob_int, log_prob_ext, log_prob_position
        ], dim=-1), torch.stack([
            entropy_int, entropy_ext, entropy_position
        ], dim=-1)


    def get_log_probability_with_aux_loss(self, context, indices, selected_action, valid_actions=None):
        # context has shape (batch, episodic_memory_size, self.packed_context_size)
        # indices has shape (batch, context_size + 1, working_memory_size) with value in [0, episodic_memory_size - 1]

        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32).to(self.device)
        if isinstance(indices, np.ndarray):
            indices = torch.tensor(indices, dtype=torch.long).to(self.device)

        indices = indices[:, :-1, :]  # remove last indices for computing logprob

        available_flags = None
        available_actions = None
        if valid_actions is not None:
            if isinstance(valid_actions, np.ndarray):
                valid_actions = torch.tensor(valid_actions, dtype=torch.bool).to(self.device)
            # valid_actions has shape (batch, context_size, int_action_size + ext_action_size)
            available_flags = valid_actions[:, :, :self.int_action_size].to(self.device)
            available_actions = valid_actions[:, :, self.int_action_size:].to(self.device)

        logits_int, logits_ext, position_logits = self.compute(context, indices)

        probs_int = Categorical_With_Mask(logits=logits_int, mask=available_flags)
        probs_ext = Categorical_With_Mask(logits=logits_ext, mask=available_actions)
        probs_position = Bernoulli(probs=position_logits)

        action_int = selected_action[:, :, 0]
        action_ext = selected_action[:, :, 1]
        action_position = selected_action[:, :, (1 + 1): (1 + 1 + self.position_size)]

        log_prob_int = probs_int.log_prob(action_int)
        log_prob_ext = probs_ext.log_prob(action_ext)
        log_prob_position = probs_position.log_prob(action_position).mean(-1)
        
        entropy_int = probs_int.entropy()
        entropy_ext = probs_ext.entropy()
        entropy_position = probs_position.entropy().mean(-1)

        return torch.stack([
            log_prob_int, log_prob_ext, log_prob_position
        ], dim=-1), torch.stack([
            entropy_int, entropy_ext, entropy_position
        ], dim=-1), None  # no aux loss for now


    def unpack_action(self, packed_action):
        # packed_action has shape (batch, self.packed_action_size)
        # return int_action, ext_action, position

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
class Projector:
    def __init__(self, master_core, selected_indices=[0, 1, 2]):
        self.master_core = master_core
        self.selected_indices = selected_indices

    def parameters(self):
        return self.master_core.parameters()
    
    def get_log_probability(self, context, indices, selected_action, valid_actions=None):
        all_logprobs, all_entropy = self.master_core.get_log_probability(context, indices, selected_action, valid_actions)
        log_probs = all_logprobs[:, :, self.selected_indices].sum(dim=-1)  # sum over selected logprob components
        entropy = all_entropy[:, :, self.selected_indices].sum(dim=-1)
        return log_probs, entropy
    
    def get_log_probability_with_aux_loss(self, context, indices, selected_action, valid_actions=None):
        all_logprobs, all_entropy, svl_unsum_loss = self.master_core.get_log_probability_with_aux_loss(context, indices, selected_action, valid_actions)
        log_probs = all_logprobs[:, :, self.selected_indices].sum(dim=-1)  # sum over selected logprob components
        entropy = all_entropy[:, :, self.selected_indices].sum(dim=-1)
        return log_probs, entropy, svl_unsum_loss
    
    def train(self):
        self.master_core.train()

    def eval(self):
        self.master_core.eval()
