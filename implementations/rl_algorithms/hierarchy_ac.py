from typing import List, Mapping, Any, Optional
from collections import defaultdict
import numpy as np
import random
import os
import logging

from implementations.core.torch.multigoal_tf import Model
from .base import Hierarchy_Base, Network_Scale_Preset, Value_Action
from implementations.core.torch.base import Log_Softmax_Function, Exp_Entropy_Function, softmax_with_temperature

import torch
import torch.nn as nn
from torch import optim

torch.autograd.set_detect_anomaly(False)


class Hierarchy_AC(Hierarchy_Base):

    def __init__(self, input_size, network_preset: Network_Scale_Preset, device, discount_factor=0.99, learning_rate=0.0001, entropy_weight=0.1, train_temperature=1.0) -> None:
        
        if network_preset == Network_Scale_Preset.small:
            model = Model(
                input_size=input_size, hidden_size=64,
                context_head=16, context_layers=2,
                objective_head=8, objective_layers=2,
                action_head=8, action_layers=2,
                value_head=16, value_layers=4,
                policy_head=16, policy_layers=4,
                device=device)
        elif network_preset == Network_Scale_Preset.medium:
            model = Model(
                input_size=input_size, hidden_size=128,
                context_head=16, context_layers=4,
                objective_head=8, objective_layers=2,
                action_head=8, action_layers=2,
                value_head=16, value_layers=8,
                policy_head=16, policy_layers=8,
                device=device)
        elif network_preset == Network_Scale_Preset.large:
            model = Model(
                input_size=input_size, hidden_size=256,
                context_head=32, context_layers=4,
                objective_head=16, objective_layers=2,
                action_head=16, action_layers=2,
                value_head=32, value_layers=12,
                policy_head=32, policy_layers=12,
                device=device)
        
        optimizer = optim.Adam(model.parameters(), learning_rate)
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=10000)
        self.learning_rate = learning_rate
        self.entropy_weight = entropy_weight
        super().__init__(model=model, optimizer=optimizer, scheduler=scheduler, device=device, discount_factor=discount_factor, train_temperature=train_temperature)


    def reset(self):
        self.model.reset_parameters()
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        super().reset()


    def act(self, objective_tensor: Any, state_tensor: Any, action_list_tensor: Any, action_list: List[str], sample_action=True):
        # action_list_tensor has shape (all_action_length, action_size)
        with torch.no_grad():
            action_scores = self._compute_action_scores(objective_tensor, state_tensor, action_list_tensor)
            if sample_action:
                # sample
                probs = softmax_with_temperature(action_scores, temperature=self.train_temperature, dim=0) # n_actions
                index = torch.multinomial(probs, num_samples=1).item() # 1
                rank = torch.argsort(action_scores, descending=True).tolist().index(index) + 1
            else:
                # greedy
                index = torch.argmax(action_scores, dim=0).item()
                rank = 1
        return Value_Action(action_list[index], rank, self.iteration)


    def train(self, train_last_node, pivot: List[Any], train_data: List[Any], objective_tensor:Any, state_tensor: Any, action_list_tensor: Any, action_list: List[str]):
        # pivot is a list of tuples (you will get this reward, moving from this context index, with the following available actions), must be sorted
        # train_data is a list of tuples (you use this action, to go from this pivot, to this pivot)

        context_size = state_tensor.size(0)
        action_size = action_list_tensor.size(1)
        num_pivot = len(pivot)

        # ----------------------
        # first prepare pivots

        # compute values of all context steps
        objective_tensor = torch.reshape(objective_tensor, [1, -1])
        state_tensor = torch.reshape(state_tensor, [1, -1, state_tensor.size(1)])

        context_marks = torch.tensor([m for m, _ in pivot], dtype=torch.int64, device=self.device)
        
        # now prepare available actions
        max_available_actions = max([len(aa) for _, aa in pivot])
        available_actions_indices = []
        padded_available_actions_by_pivot = []
        action_set = set(action_list)
        for _, aa in pivot:
            # compute free actions
            free_actions = action_set - set(aa)
            new_aa = list(aa)
            # now add missing
            while len(new_aa) < max_available_actions:
                new_aa.append(random.choice(list(free_actions)))
            available_actions_indices.append([action_list.index(a) for a in new_aa])
            padded_available_actions_by_pivot.append(new_aa)
        available_actions_indices = torch.tensor(available_actions_indices, dtype=torch.int64, device=self.device) # shape: (num_pivot, max_action_length)
        
        # action_list_tensor has shape (all_action_length, action_size) must be expanded to (num_pivot, all_action_length, action_size)
        # available_actions_indices has shape (num_pivot, max_action_length) must be expanded to (num_pivot, max_action_length, action_size)
        available_actions_by_context = torch.gather(action_list_tensor.unsqueeze(0).expand(num_pivot, -1, -1), 1, available_actions_indices.unsqueeze(2).expand(-1, -1, action_size)) # shape: (num_pivot, max_action_length, action_size)
        available_actions_by_context = torch.reshape(available_actions_by_context, [1, num_pivot, max_available_actions, action_size])
        
        self.model.train()
        action_scores, values = self.model(objective_tensor, state_tensor, available_actions_by_context, torch.reshape(context_marks, (1, -1)))
        action_scores = action_scores[0, :, :] # shape: (num_pivot, max_available_actions)
        state_values = values[0, :] # shape: (num_pivot)

        # ----------------------
        # now map to training data items
        train_action_indexes = torch.reshape(torch.tensor([padded_available_actions_by_pivot[p].index(a) for _, a, p, _ in train_data], dtype=torch.int64, device=self.device), (-1, 1))
        train_from_indexes = torch.tensor([p for _, _, p, _ in train_data], dtype=torch.int64, device=self.device)
        train_to_indexes = torch.tensor([p for _, _, _, p in train_data], dtype=torch.int64, device=self.device)
        train_action_scores = torch.gather(action_scores, 0, train_from_indexes.unsqueeze(-1).expand(-1, max_available_actions))
        train_rewards = torch.tensor([r for r, _, _, _ in train_data], dtype=torch.float32, device=self.device)

        train_state_values = torch.gather(state_values, 0, train_from_indexes)

        with torch.no_grad():
            # now specialize truncated end
            if not train_last_node:
                last_value = state_values[-1].item()
            else:
                last_value = 0

            train_mc_returns = self._compute_discounted_mc_returns(train_rewards, last_value)
            train_advantages = train_mc_returns - train_state_values

        # use vector instead of loops
        log_probs = Log_Softmax_Function.apply(train_action_scores, self.train_temperature, 1)
        log_action_probs = torch.gather(log_probs, 1, train_action_indexes)
        log_action_probs = log_action_probs.flatten()
        policy_loss = (-log_action_probs * train_advantages).sum()
        value_loss = (.5 * (train_state_values - train_mc_returns) ** 2.).sum()
        entropy = Exp_Entropy_Function.apply(log_probs, 1).sum()  # Use custom entropy function for stability
        loss = policy_loss + 0.5 * value_loss - self.entropy_weight * entropy # entropy has to be adjusted, too low and it will get stuck at a command.

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.iteration += 1

        self.ave_loss = self.LOG_ALPHA * self.ave_loss + (1 - self.LOG_ALPHA) * loss.item()
