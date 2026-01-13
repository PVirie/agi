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
        self.num_minibatches = 8

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


    def train(self, obs: Any, actions: Any, target_actions: Any, valid_actions: Any = None, masks: Any = None):
        """
        obs: np array of shape (batch_size, context_length, ...)
        actions: np array of shape (batch_size, context_length, ...)
        target_actions: np array of shape (batch_size, context_length, ...)
        valid_actions: np array of shape (batch_size, context_length, ...)
        masks: np array of shape (batch_size, context_length)
        """
        b_obs = convert_np_array_to_float_tensor(obs, self.device)
        b_actions = convert_np_array_to_float_tensor(actions, self.device)
        b_target_actions = convert_np_array_to_float_tensor(target_actions, self.device)
        b_valid_actions = convert_np_array_to_bool_tensor(valid_actions, self.device) if valid_actions is not None else None
        b_masks = convert_np_array_to_float_tensor(masks, self.device) if masks is not None else None

        batch_size = b_actions.shape[0]
        sequence_size = b_actions.shape[1]
        minibatch_size = min(batch_size // self.num_minibatches, 8)

        for epoch in range(self.update_epochs):
            # logprobs, _ = self.policy_model.get_log_probability(
            #     context=obs, 
            #     action=actions,
            #     valid_actions=valid_actions,
            #     target_action=target_actions
            # )
            # the last section consume too much v_ram, need to split into minibatches
            logprobs_list = []
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_obs = b_obs[start:end, ...]
                mb_actions = b_actions[start:end, ...]
                mb_valid_actions = b_valid_actions[start:end, ...] if b_valid_actions is not None else None
                mb_target_actions = b_target_actions[start:end, ...]

                mb_logprob, _ = self.policy_model.get_log_probability(
                    context=mb_obs, 
                    action=mb_actions,
                    valid_actions=mb_valid_actions,
                    target_action=mb_target_actions
                )
                logprobs_list.append(mb_logprob)
            logprobs = torch.cat(logprobs_list, dim=0)

            # now attempt to minimize negative log likelihood
            if b_masks is None:
                loss = -logprobs.mean()
            else:
                loss = -masked_mean(logprobs, b_masks)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

