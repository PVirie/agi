from typing import Any, List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging

from interfaces.learning import RL_Learner
from interfaces.network import Policy_Network, Value_Network
from interfaces.data_structure import Context_Collector
from utilities.safe_torch_module import Safe_nn_Module
from implementations.learning_algorithms.torch.ppo import PPO


class PPO_With_Rapid_Parameters(PPO):

    def __init__(self, policy_model: Policy_Network, value_model: Value_Network, device, persistence_path=None, minibatch_size=8, aux_coef=None):
        """
        aux_coef: Coefficient for auxiliary value loss. If None, no auxiliary value loss is used.
        """
        self.policy_model = policy_model
        self.value_model = value_model
        self.device = device

        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_coef = 0.2
        self.norm_adv = True
        self.clip_vloss = True
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.target_kl = None

        self.aux_coef = aux_coef

        self.update_epochs = 4
        self.minibatch_size = minibatch_size

        self.all_parameters = list(self.policy_model.slow_parameters()) + list(self.value_model.parameters())
        self.optimizer = optim.Adam([
            {'params': self.all_parameters, 'lr': self.lr}, # Background learning
            {'params': self.policy_model.fast_parameters(), 'lr': 0.1}   # Rapid adaptation
        ])

        Safe_nn_Module.__init__(self, 
            device=device, persistence_path=persistence_path, 
            modules={"ppo_learner": self.optimizer}
        )

        self.load()
