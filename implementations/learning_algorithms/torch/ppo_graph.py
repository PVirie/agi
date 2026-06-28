from typing import Any, List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging

from interfaces.learning import RL_Learner
from interfaces.network import Policy_Value_Network
from utilities.safe_torch_module import Safe_nn_Module

from .base import convert_list_of_list_of_bool_to_float_tensor, convert_np_array_to_float_tensor, convert_list_of_np_array_to_float_tensor, convert_list_of_np_array_to_int_tensor, convert_np_array_to_bool_tensor
from .base import masked_mean, masked_std


class PPO(RL_Learner, Safe_nn_Module):

    def __init__(self, 
                 policy_model: Policy_Value_Network, 
                 device, persistence_path=None, 
                 lr=3e-4, minibatch_size=8):
        """
        PPO: Proximal Policy Optimization Algorithm

        Issue: NaN when the sample actions are not on-policy.
        """
        self.policy_model = policy_model
        self.device = device

        self.lr = lr
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_coef = 0.2
        self.norm_adv = True
        self.clip_vloss = True
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.target_kl = None

        self.update_epochs = 4
        self.minibatch_size = minibatch_size

        self.all_parameters = list(self.policy_model.parameters())
        self.optimizer = optim.Adam(self.all_parameters, lr=self.lr, eps=1e-5)

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
              obs: Any, 
              last_actions: Any, 
              rewards: List[Any], 
              next_dones: List[List[bool]],
              valid_actions: Any = None, 
              masks: Any = None, 
              causes: Any = None
            ):
        """
        obs: np array of shape (batch_size, context_length + 1, ...)
        last_actions: np array of shape (batch_size, context_length + 1, ...)
        rewards: list (context_length) of np array of shape (batch_size)
        next_dones: list (context_length) of bools of length batch_size
        valid_actions: np array of shape (batch_size, context_length, ...)
        masks: np array of shape (batch_size, context_length)
        causes: np array of shape (batch_size, context_length + 1, cause_size)

        Note that obs and last_actions include the context length + 1 items.
        This corresponds to the observations after taking the last actions.

        causes is a list of previous context indices that cause the current one.
        All return values from the current context index should be propagated to the cause context indices as well, not just the previous step.
        """
        # Use dim 0 as context length dimension
        b_obs = convert_np_array_to_float_tensor(obs, self.device)
        b_actions = convert_np_array_to_float_tensor(last_actions, self.device)
        b_rewards = convert_list_of_np_array_to_float_tensor(rewards, self.device)
        b_next_dones = convert_list_of_list_of_bool_to_float_tensor(next_dones, self.device)
        b_valid_actions = convert_np_array_to_bool_tensor(valid_actions, self.device) if valid_actions is not None else None
        b_masks = torch.ones_like(b_rewards).to(self.device) if masks is None else convert_np_array_to_float_tensor(masks, self.device)
        b_causes = convert_list_of_np_array_to_int_tensor(causes, self.device) if causes is not None else None

        batch_size = b_rewards.shape[0]
        sequence_size = b_rewards.shape[1]

        self.policy_model.eval()

        with torch.no_grad():
            logprobs_list = []
            values_with_last_list = []
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_obs = b_obs[start:end, ...]
                mb_actions = b_actions[start:end, ...]
                mb_valid_actions = b_valid_actions[start:end, ...] if b_valid_actions is not None else None

                mb_logprob, _, mb_values = self.policy_model.get_log_probability_with_value(
                    context=mb_obs,
                    selected_action=mb_actions[:, 1:, ...],
                    valid_actions=mb_valid_actions,
                )
                logprobs_list.append(mb_logprob)

                values_with_last_list.append(mb_values)

            logprobs = torch.cat(logprobs_list, dim=0)
            values_with_last = torch.cat(values_with_last_list, dim=0)

            values = values_with_last[:, :-1, ...]  # (batch_size, context_length)

            # Bootstrap value using standard GAE, with causal credit assignment merged into the
            # same backward pass.  causal_returns[b, c] accumulates the max return seen from any
            # future step that c caused.  Because causes always point backward (c < t), by the
            # time the loop reaches c every future t > c has already propagated its credit, so
            # causal_returns[:, c] is fully populated when it is consumed.
            causal_returns = torch.full((batch_size, sequence_size), float('-inf'), device=self.device) if b_causes is not None else None
            advantages = []
            lastgaelam = torch.zeros(batch_size).to(self.device)
            for t in reversed(range(sequence_size)):
                nextnonterminal = 1.0 - b_next_dones[:, t]
                nextvalues = values_with_last[:, t + 1]
                delta = b_rewards[:, t] + self.gamma * nextvalues * nextnonterminal - values_with_last[:, t]
                lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam

                standard_return = lastgaelam + values[:, t]             # (batch_size,)

                # Take the best of the standard return and any causal credit received from
                # future steps, then propagate that effective return back to all past causes
                # of t.  Propagating the effective (post-max) value means transitive chains
                # (c caused t, t caused t') are handled without an extra pass.
                if causal_returns is not None:
                    effective_return = torch.max(standard_return, causal_returns[:, t])
                    # Vectorised scatter-max: propagate effective_return[b] to every valid
                    # cause position c for each batch item b, in one operation.
                    all_causes = b_causes[:, t, :]                              # (batch_size, cause_size)
                    valid_mask = (all_causes >= 0) & (all_causes < sequence_size)
                    safe_causes = all_causes.clamp(0, sequence_size - 1)        # prevent OOB on scatter
                    src = effective_return[:, None].expand(-1, b_causes.shape[2])   # (batch_size, cause_size)
                    src = src.masked_fill(~valid_mask, float('-inf'))            # invalid slots → -inf (no-op under max)
                    causal_returns = causal_returns.scatter_reduce(1, safe_causes, src, reduce='amax', include_self=True)
                else:
                    effective_return = standard_return

                advantages.append(effective_return - values[:, t])

            advantages.reverse()
            advantages = torch.stack(advantages, dim=1)
            returns = advantages + values

            if self.norm_adv:
                adv_mean = masked_mean(advantages, b_masks)
                adv_std = masked_std(advantages, b_masks)
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        self.policy_model.train()

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

                mb_obs = b_obs[mb_inds, ...]
                mb_actions = b_actions[mb_inds, 1:, ...]
                mb_valid_actions = b_valid_actions[mb_inds, ...] if b_valid_actions is not None else None
                
                mb_newlogprob, mb_entropy, mb_newvalue = self.policy_model.get_log_probability_with_value(
                    context=mb_obs,
                    selected_action=mb_actions,
                    valid_actions=mb_valid_actions,
                )
                mb_newlogprob = mb_newlogprob
                mb_newvalue = mb_newvalue[:, :-1, ...]  # remove last value
                
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

                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.all_parameters, self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        self.policy_model.eval()

