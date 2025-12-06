from typing import Any, List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging

from interfaces.learning import Supervised_Learner
from interfaces.core import Core

from .base import convert_np_array_to_float_tensor, masked_mean

class Basic_Learner(Supervised_Learner):

    def __init__(self, agent: Core, device, persistence_path=None):
        self.agent = agent
        self.device = device

        self.lr = 3e-4
        self.max_grad_norm = 0.5

        self.update_epochs = 10
        self.num_minibatches = 32

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr, eps=1e-5)
        
        self.persistence_path = persistence_path
        if self.persistence_path is not None:
            self.load()


    def reset(self, time = 0.0):
        frac = 1.0 - time
        lrnow = frac * self.lr
        self.optimizer.param_groups[0]["lr"] = lrnow


    def load(self):
        if self.persistence_path is not None:
            try:
                checkpoint = torch.load(f"{self.persistence_path}/basic_learner_checkpoint.pth", map_location=self.device)
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logging.info(f"Loaded Basic Learner from {self.persistence_path}/basic_learner_checkpoint.pth")
            except FileNotFoundError:
                logging.info(f"No Basic Learner checkpoint found at {self.persistence_path}/basic_learner_checkpoint.pth, starting fresh.")
    
    
    def save(self):
        if self.persistence_path is not None:
            torch.save({
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, f"{self.persistence_path}/basic_learner_checkpoint.pth")
            logging.info(f"Saved Basic Learner to {self.persistence_path}/basic_learner_checkpoint.pth")


    def train(self, obs: Any, target_actions: Any, masks: Any = None):
        obs = convert_np_array_to_float_tensor(obs, self.device)
        target_actions = convert_np_array_to_float_tensor(target_actions, self.device)

        batch_size = target_actions.shape[0]
        sequence_size = target_actions.shape[1]
        minibatch_size = max(sequence_size // self.num_minibatches, 8)

        for epoch in range(self.update_epochs):
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size

                mb_target_actions = target_actions[:, start:end]
                mb_masks = masks[:, start:end, ...]

                if torch.sum(mb_masks) < 1e-8:
                    continue

                _, _, b_newlogprob, _, _ = self.agent.get_action_and_value(
                    obs, 
                    target_actions,
                    use_action=True,
                    use_grad=True
                )

                # now attempt to minimize negative log likelihood


                # self.optimizer.zero_grad()
                # loss.backward()
                # nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                # self.optimizer.step()

