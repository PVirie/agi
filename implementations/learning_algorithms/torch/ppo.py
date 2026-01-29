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

from .base import convert_list_of_list_of_bool_to_float_tensor, convert_np_array_to_float_tensor, convert_list_of_np_array_to_float_tensor, convert_np_array_to_bool_tensor
from .base import masked_mean, masked_std


class PPO(RL_Learner, Safe_nn_Module):

    def __init__(self, policy_model: Policy_Network, value_model: Value_Network, device, persistence_path=None, minibatch_size=8, svl_coef=None):
        """
        svl_coef: Coefficient for supervised value loss. If None, no supervised value loss is used.
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

        self.svl_coef = svl_coef

        self.update_epochs = 4
        self.minibatch_size = minibatch_size

        all_parameters = list(self.policy_model.parameters()) + list(self.value_model.parameters())
        self.optimizer = optim.Adam(all_parameters, lr=self.lr, eps=1e-5)

        Safe_nn_Module.__init__(self, 
            device=device, persistence_path=persistence_path, 
            modules={"ppo_learner": self.optimizer}
        )

        self.load()


    def update_learning_rate(self, time = 0.0):
        frac = max(1.0 - time, 0.01)
        lrnow = frac * self.lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lrnow


    def learn(self, 
              obs: Any, last_actions: Any, rewards: List[Any], 
              next_dones: List[List[bool]],
              valid_actions: Any = None, masks: Any = None, svl_masks: Any = None):
        """
        obs: np array of shape (batch_size, context_length + 1, ...)
        last_actions: np array of shape (batch_size, context_length + 1, ...)
        rewards: list (context_length) of np array of shape (batch_size)
        next_dones: list (context_length) of bools of length batch_size
        valid_actions: np array of shape (batch_size, context_length, ...)
        masks: np array of shape (batch_size, context_length)
        svl_masks: np array of shape (batch_size, context_length)

        Note that obs and last_actions include the context length + 1 items.
        This corresponds to the observations after taking the last actions.
        """
        # Use dim 0 as context length dimension
        b_obs = convert_np_array_to_float_tensor(obs, self.device)
        b_actions = convert_np_array_to_float_tensor(last_actions, self.device)
        b_rewards = convert_list_of_np_array_to_float_tensor(rewards, self.device)
        b_next_dones = convert_list_of_list_of_bool_to_float_tensor(next_dones, self.device)
        b_valid_actions = convert_np_array_to_bool_tensor(valid_actions, self.device) if valid_actions is not None else None
        b_masks = torch.ones_like(b_rewards).to(self.device) if masks is None else convert_np_array_to_float_tensor(masks, self.device)
        
        if self.svl_coef is not None:
            b_svl_masks = torch.ones_like(b_rewards).to(self.device) if svl_masks is None else convert_np_array_to_float_tensor(svl_masks, self.device)

        batch_size = b_rewards.shape[0]
        sequence_size = b_rewards.shape[1]

        self.policy_model.eval()
        self.value_model.eval()

        with torch.no_grad():
            logprobs_list = []
            values_with_last_list = []
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_obs = b_obs[start:end, ...]
                mb_actions = b_actions[start:end, ...]
                mb_valid_actions = b_valid_actions[start:end, ...] if b_valid_actions is not None else None

                mb_logprob, _ = self.policy_model.get_log_probability(
                    context=mb_obs[:, :-1, ...],
                    selected_action=mb_actions[:, 1:, ...],
                    valid_actions=mb_valid_actions,
                )
                logprobs_list.append(mb_logprob)

                mb_values = self.value_model.get_value(mb_obs)
                values_with_last_list.append(mb_values)

            logprobs = torch.cat(logprobs_list, dim=0)
            values_with_last = torch.cat(values_with_last_list, dim=0)

            values = values_with_last[:, :-1, ...]  # (batch_size, context_length)

            # Bootstrap value if not done
            advantages = []
            lastgaelam = torch.zeros(batch_size).to(self.device)
            for t in reversed(range(sequence_size)):
                nextnonterminal = 1.0 - b_next_dones[:, t]
                nextvalues = values_with_last[:, t + 1]
                delta = b_rewards[:, t] + self.gamma * nextvalues * nextnonterminal - values_with_last[:, t]
                lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                advantages.append(lastgaelam)
            advantages.reverse()
            advantages = torch.stack(advantages, dim=1)
            returns = advantages + values

            if self.norm_adv:
                adv_mean = masked_mean(advantages, b_masks)
                adv_std = masked_std(advantages, b_masks)
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        self.policy_model.train()
        self.value_model.train()

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                mb_log_prob = logprobs[mb_inds, ...]
                mb_value = values[mb_inds, ...]
                mb_advantages = advantages[mb_inds, ...] 
                mb_returns = returns[mb_inds, ...]
                mb_masks = b_masks[mb_inds, ...]

                if torch.sum(mb_masks) < 1e-8:
                    continue

                if self.svl_coef is not None:
                    mb_obs = b_obs[mb_inds, :, ...]
                    mb_actions = b_actions[mb_inds, 1:, ...]
                    mb_valid_actions = b_valid_actions[mb_inds, ...] if b_valid_actions is not None else None
                    
                    mb_newlogprob, mb_entropy, svl_unsum_loss = self.policy_model.get_log_probability_with_svl_loss(
                        context=mb_obs,
                        selected_action=mb_actions,
                        valid_actions=mb_valid_actions
                    )
                    svl_loss = masked_mean(svl_unsum_loss, b_svl_masks[mb_inds, ...])

                    mb_newvalue = self.value_model.get_value(mb_obs[:, :-1, ...])
                else:
                    mb_obs = b_obs[mb_inds, :-1, ...]
                    mb_actions = b_actions[mb_inds, 1:, ...]
                    mb_valid_actions = b_valid_actions[mb_inds, ...] if b_valid_actions is not None else None
                    
                    mb_newlogprob, mb_entropy = self.policy_model.get_log_probability(
                        context=mb_obs,
                        selected_action=mb_actions,
                        valid_actions=mb_valid_actions,
                    )
                    mb_newvalue = self.value_model.get_value(mb_obs)
                
                logratio = mb_newlogprob - mb_log_prob
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = masked_mean(-logratio, mb_masks)
                    approx_kl = masked_mean((ratio - 1) - logratio, mb_masks)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = masked_mean(torch.max(pg_loss1, pg_loss2), mb_masks)

                # Value loss
                if self.clip_vloss:
                    v_loss_unclipped = (mb_newvalue - mb_returns) ** 2
                    v_clipped = mb_value + torch.clamp(
                        mb_newvalue - mb_value,
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * masked_mean(v_loss_max, mb_masks)
                else:
                    v_loss = 0.5 * masked_mean((mb_newvalue - mb_returns) ** 2, mb_masks)

                entropy_loss = masked_mean(mb_entropy, mb_masks)

                if self.svl_coef is not None:
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef + svl_loss * self.svl_coef
                else:
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                all_parameters = list(self.policy_model.parameters()) + list(self.value_model.parameters())
                nn.utils.clip_grad_norm_(all_parameters, self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        self.policy_model.eval()
        self.value_model.eval()

