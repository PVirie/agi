from typing import List, Mapping, Any, Optional
import numpy as np
import os
import logging
from enum import Enum

import torch
import torch.nn as nn
from torch import optim

from implementations.core.torch.base import softmax_with_temperature

torch.autograd.set_detect_anomaly(False)


class Network_Scale_Preset(str, Enum):
    small = "small"
    medium = "medium"
    large = "large"


class Value_Action:
    def __init__(self, selected_action, selected_rank:int = -1, iteration=0):
        self.mdp_score = None
        self.selected_action = selected_action
        self.selected_action_rank = selected_rank
        self.available_actions = None

        self.has_released = False
        self.iteration = iteration

    def release(self):
        self.has_released = True


class Hierarchy_Base:
    MAX_CONTEXT_SIZE = 512

    def __init__(self, model, optimizer, scheduler, device, discount_factor=0.99, log_alpha=0.95, train_temperature=1.0):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.GAMMA = discount_factor
        self.LOG_ALPHA = log_alpha
        self.train_temperature = train_temperature

        self.ave_loss = 0
        self.iteration = 0
        self.not_first_reset = False

        # use mesh to build gammas
        x = torch.arange(0, self.MAX_CONTEXT_SIZE, dtype=torch.float32, device=device)
        y = torch.arange(0, -self.MAX_CONTEXT_SIZE, -1, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        grid_xy = grid_x + grid_y
        self.gammas = torch.pow(self.GAMMA, grid_xy).to(device)
        # now only keep top right triangle
        self.gammas = torch.triu(self.gammas, diagonal=0)


    def reset(self):
        self.ave_loss = 0
        self.iteration = 0
        logging.info("Model reset.")


    def save(self, dir_path):
        torch.save(self.model.state_dict(), os.path.join(dir_path, "model.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(dir_path, "optimizer.pth"))
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(dir_path, "scheduler.pth"))
        torch.save({
            'iteration': self.iteration,
            'ave_loss': self.ave_loss,
        }, os.path.join(dir_path, "state.pth"))


    def load(self, dir_path):
        if not os.path.exists(os.path.join(dir_path, "state.pth")):
            return False
        self.model.load_state_dict(torch.load(os.path.join(dir_path, "model.pth"), map_location=self.device))
        self.optimizer.load_state_dict(torch.load(os.path.join(dir_path, "optimizer.pth"), map_location=self.device))
        if os.path.exists(os.path.join(dir_path, "scheduler.pth")):
            self.scheduler.load_state_dict(torch.load(os.path.join(dir_path, "scheduler.pth"), map_location=self.device))
        state = torch.load(os.path.join(dir_path, "state.pth"))
        self.iteration = state['iteration']
        self.ave_loss = state['ave_loss']
        return True


    def print(self, step):
        msg = "{:6d}. ".format(step)
        msg += "loss: {:5.2f}; ".format(self.ave_loss)
        logging.info(msg)


    def _compute_discounted_mc_returns(self, rewards, last_value):
        context_length = rewards.size(0)
        last_value = last_value if isinstance(last_value, torch.Tensor) else torch.ones(1, device=self.device) * last_value
        # append last value to rewards
        R = torch.cat((rewards, last_value), dim=0)
        # gammas is [[r^0, r^1, ..., r^n], [0, r^0, r^1, ..., r^n-1], ...]
        # S_i = R_i + gamma*(S_i+1)
        S = torch.matmul(self.gammas[:context_length, :context_length + 1], R)
        return S


    def _compute_snake_ladder(self, rewards, last_value):
        # this is not true snake ladder, but a simplified version
        # return a 2D matrix that contains best return from step i to step j
        # rewards is a 1D tensor
        # last_value is a scalar

        # use vector instead of loops
        context_length = rewards.size(0)
        last_value = last_value if isinstance(last_value, torch.Tensor) else torch.ones(1, device=self.device) * last_value
        
        # append last value to rewards
        R = torch.cat((rewards, last_value), dim=0)
        # gammas is [[r^0, r^1, ..., r^n], [0, r^0, r^1, ..., r^n-1], ...]
        # S_i = R_i + gamma*(S_i+1)
        S = torch.matmul(self.gammas[:context_length, :context_length + 1], R)
        
        # cumulative sum of R = [R[0], R[0] + R[1], R[0] + R[1] + R[2], ...]
        cR = torch.cumsum(R, dim=0)
        # now create from to sum table of R; i.e. K[i, j] = R[i] + R[i+1] + ... + R[j - 2] = cR[j - 2] - cR[i - 1]
        # use grid difference of cR
        grid_i, grid_j = torch.meshgrid(torch.arange(0, context_length, device=self.device), torch.arange(0, context_length + 1, device=self.device), indexing='ij')
        grid_ji = grid_j - grid_i
        K = torch.where(grid_ji > 0, cR[grid_j - 2] - cR[grid_i - 1], 0)

        # now we make final matrix W[i, j] = K[i, j] + S[j - 1]
        W = K + torch.where(grid_ji > 0, S[grid_j - 1], 0)

        return W
    

    def _compute_snake_ladder_2(self, rewards, state_q):
        # return a 2D matrix that contains best return from step i to step j
        # rewards is a 1D tensor of size (context_length)
        # state_q is a 1D tensor of size (context_length + 1)
        # return a 2D matrix of size (rewards.size(0), rewards.size(0) + 1) each element computes a return if you take action i and then go to j
        # just compute the sum of reward return[i, j] = r[i] + r[i+1] + ... + r[j-1] + self.gamma * state_q[j]
        context_length = rewards.size(0)
        cR = torch.cumsum(rewards, dim=0)
        grid_i, grid_j = torch.meshgrid(torch.arange(0, context_length, device=self.device), torch.arange(0, context_length + 1, device=self.device), indexing='ij')
        grid_ji = grid_j - grid_i
        K = torch.where(grid_ji > 0, cR[grid_j - 1] - cR[grid_i - 1], 0)
        W = K + self.GAMMA * torch.reshape(state_q, (1, -1))
        return W
    

    def _compute_action_scores(self, objective_tensor: Any, state_tensor: Any, action_list_tensor: Any):
        # action_list_tensor has shape (all_action_length, action_size)
        
        n_context = state_tensor.size(0)
        objective_tensor = torch.reshape(objective_tensor, [1, -1])
        state_tensor = torch.reshape(state_tensor, [1, -1, state_tensor.size(1)])
        action_list_tensor = torch.reshape(action_list_tensor, [1, 1, -1, action_list_tensor.size(1)])
        pivot_positions = torch.tensor([[n_context - 1]], dtype=torch.int64, device=self.device) # shape: (1, 1)

        self.model.eval()
        action_scores, _ = self.model(objective_tensor, state_tensor, action_list_tensor, pivot_positions)
        action_scores = action_scores[0, 0, :]

        return action_scores


    def act(self, objective_tensor: Any, state_tensor: Any, action_list_tensor: Any, action_list: List[str], sample_action=True):
        raise NotImplementedError("act() is not implemented in base class")
    
    
    def train(self, train_last_node, pivot: List[Any], train_data: List[Any], objective_tensor:Any, state_tensor: Any, action_list_tensor: Any, action_list: List[str]):
        raise NotImplementedError("train() is not implemented in base class")


    def update_sheduler(self):
        if self.scheduler is not None and self.not_first_reset:
            self.scheduler.last_epoch = -1
            self.scheduler.step()
            self.not_first_reset = True