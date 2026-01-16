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

    def __init__(self, policy_model: Policy_Network, device, persistence_path=None, minibatch_size=8):
        self.policy_model = policy_model
        self.device = device

        self.lr = 3e-4
        self.max_grad_norm = 0.5

        self.update_epochs = 1
        self.minibatch_size = minibatch_size

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


    def train(self, obs: Any, actions: Any, valid_actions: Any = None, masks: Any = None):
        """
        obs: np array of shape (batch_size, context_length, ...)
        actions: np array of shape (batch_size, context_length + 1, ...)
        valid_actions: np array of shape (batch_size, context_length, ...)
        masks: np array of shape (batch_size, context_length)
        """
        batch_size = actions.shape[0]
        sequence_size = actions.shape[1]

        b_obs = convert_np_array_to_float_tensor(obs, self.device)
        b_actions = convert_np_array_to_float_tensor(actions, self.device)
        b_valid_actions = convert_np_array_to_bool_tensor(valid_actions, self.device) if valid_actions is not None else None
        b_masks = torch.ones(batch_size, sequence_size).to(self.device) if masks is None else convert_np_array_to_float_tensor(masks, self.device)

        b_inds = np.arange(batch_size)
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs = b_obs[mb_inds, ...]
                mb_actions = b_actions[mb_inds, ...]
                mb_valid_actions = b_valid_actions[mb_inds, ...] if b_valid_actions is not None else None
                mb_masks = b_masks[mb_inds, ...]

                if torch.sum(mb_masks) < 1e-8:
                    continue

                mb_logprob, _ = self.policy_model.get_log_probability(
                    context=mb_obs, 
                    action=mb_actions[:, :-1, ...],
                    valid_actions=mb_valid_actions,
                    target_action=mb_actions[:, 1:, ...]
                )

                # now attempt to minimize negative log likelihood
                loss = -masked_mean(mb_logprob, mb_masks)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
                self.optimizer.step()

