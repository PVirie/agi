from typing import Any, List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging

from interfaces.learning import RL_Learner
from interfaces.network import Policy_Network
from interfaces.data_structure import Context_Collector
from utilities.safe_torch_module import Safe_nn_Module

from .base import convert_list_of_list_of_bool_to_float_tensor, convert_np_array_to_float_tensor, convert_list_of_np_array_to_float_tensor, convert_np_array_to_bool_tensor
from .base import masked_mean, masked_std


class SAC(RL_Learner, Safe_nn_Module):

    def __init__(self, policy_model: Policy_Network, device, persistence_path=None, minibatch_size=8):
        self.policy_model = policy_model
        self.device = device

        self.lr = 3e-4
        self.gamma = 0.99
        self.tau = 0.005
        self.learning_starts = 5000
        self.policy_lr = 3e-4
        self.q_lr = 1e-3
        self.policy_frequency = 2
        self.target_network_frequency = 1
        self.alpha = 0.2
        self.autotune = True

        self.minibatch_size = minibatch_size

        Safe_nn_Module.__init__(self, name="sac_learner", device=device, persistence_path=persistence_path, module=self.optimizer)

        self.load()


    def reset(self, time = 0.0):
        frac = 1.0 - time
        lrnow = frac * self.lr
        self.optimizer.param_groups[0]["lr"] = lrnow


    def learn(self, 
              obs: Any, actions: Any, rewards: List[Any], 
              next_dones: List[List[bool]],
              valid_actions: Any = None, masks: Any = None):
        """
        obs: np array of shape (batch_size, context_length, ...)
        actions: np array of shape (batch_size, context_length, ...)
        rewards: list of np array of shape (batch_size)
        next_dones: list of bools of length batch_size
        valid_actions: np array of shape (batch_size, context_length, ...)
        masks: np array of shape (batch_size, context_length)
        """
        # Use dim 0 as context length dimension
        b_obs = convert_np_array_to_float_tensor(obs[:, :-1, ...], self.device)
        b_obs_last = convert_np_array_to_float_tensor(obs[:, -1:, ...], self.device)
        b_actions = convert_np_array_to_float_tensor(actions, self.device)
        b_rewards = convert_list_of_np_array_to_float_tensor(rewards, self.device)
        b_next_dones = convert_list_of_list_of_bool_to_float_tensor(next_dones, self.device)
        b_valid_actions = convert_np_array_to_bool_tensor(valid_actions, self.device) if valid_actions is not None else None
        b_masks = torch.ones_like(b_rewards).to(self.device) if masks is None else convert_np_array_to_float_tensor(masks, self.device)

        batch_size = actions.shape[0]
        sequence_size = actions.shape[1]



