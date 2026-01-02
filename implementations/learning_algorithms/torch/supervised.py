from typing import Any, List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging

from interfaces.learning import Supervised_Learner
from interfaces.core import Core
from utilities.safe_torch_module import Safe_nn_Module

from .base import convert_np_array_to_float_tensor, masked_mean

class Basic_Learner(Supervised_Learner, Safe_nn_Module):

    def __init__(self, agent: Core, device, persistence_path=None):
        self.agent = agent
        self.device = device

        self.lr = 3e-4
        self.max_grad_norm = 0.5

        self.update_epochs = 1

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr, eps=1e-5)
        
        Safe_nn_Module.__init__(self, name="basic_learner", device=device, persistence_path=persistence_path, module=self.optimizer)

        self.load()


    def reset(self, time = 0.0):
        frac = 1.0 - time
        lrnow = frac * self.lr
        self.optimizer.param_groups[0]["lr"] = lrnow


    def train(self, obs: Any, actions: Any, target_actions: Any, masks: Any = None):
        obs = convert_np_array_to_float_tensor(obs, self.device)
        actions = convert_np_array_to_float_tensor(actions, self.device)
        target_actions = convert_np_array_to_float_tensor(target_actions, self.device)

        for epoch in range(self.update_epochs):
            newlogprob = self.agent.get_log_probability(
                context=obs, 
                action=actions,
                target_action=target_actions,
                f_mask=masks
            )

            # now attempt to minimize negative log likelihood
            loss = -newlogprob.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optimizer.step()

