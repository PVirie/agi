from typing import Any, List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging

from interfaces.learning import Supervised_Learner
from interfaces.network import Policy_Network
from utilities.safe_torch_module import Safe_nn_Module

from .base import convert_np_array_to_float_tensor, convert_np_array_to_bool_tensor
from .base import masked_mean

class Basic_Learner(Supervised_Learner, Safe_nn_Module):

    def __init__(self, policy_model: Policy_Network, device, persistence_path=None):
        self.policy_model = policy_model
        self.device = device

        self.lr = 3e-4
        self.max_grad_norm = 0.5

        self.update_epochs = 1

        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=self.lr, eps=1e-5)
        
        Safe_nn_Module.__init__(self, 
            device=device, persistence_path=persistence_path, 
            modules={"basic_learner": self.optimizer}
        )

        self.load()


    def reset(self, time = 0.0):
        frac = 1.0 - time
        lrnow = frac * self.lr
        self.optimizer.param_groups[0]["lr"] = lrnow


    def train(self, obs: Any, actions: Any, target_actions: Any, valid_actions: Any = None, masks: Any = None, trained_logprob_indices: List[int] = None):
        """
        obs: np array of shape (batch_size, context_length, ...)
        actions: np array of shape (batch_size, context_length, ...)
        target_actions: np array of shape (batch_size, context_length, ...)
        valid_actions: np array of shape (batch_size, context_length, ...)
        masks: np array of shape (batch_size, context_length)
        trained_logprob_indices: List of int indices to select which logprob components to train on
        """
        obs = convert_np_array_to_float_tensor(obs, self.device)
        actions = convert_np_array_to_float_tensor(actions, self.device)
        target_actions = convert_np_array_to_float_tensor(target_actions, self.device)
        valid_actions = convert_np_array_to_bool_tensor(valid_actions, self.device) if valid_actions is not None else None
        masks = convert_np_array_to_float_tensor(masks, self.device) if masks is not None else None

        for epoch in range(self.update_epochs):
            logprobs = self.policy_model.get_log_probability(
                context=obs, 
                action=actions,
                valid_actions=valid_actions,
                target_action=target_actions,
                only_logprob_components=True
            )

            if trained_logprob_indices is not None:
                # select only the logprob components we want to train on from the last dimension
                logprobs = logprobs[:, :, trained_logprob_indices]
                
            sum_log_probs = logprobs.sum(dim=-1)  # sum over logprob components

            # now attempt to minimize negative log likelihood
            if masks is None:
                loss = -sum_log_probs.mean()
            else:
                loss = -masked_mean(sum_log_probs, masks)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

