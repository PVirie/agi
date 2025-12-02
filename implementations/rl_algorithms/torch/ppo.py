from typing import Any, List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from interfaces.learning import Learner
from interfaces.core import Core, Context_Collector


def convert_list_of_bool_to_float_tensor(bool_list: List[bool], device) -> torch.Tensor:
    return torch.tensor([1.0 if b else 0.0 for b in bool_list], dtype=torch.float32).to(device)


def convert_list_of_float_to_float_tensor(float_list: List[float], device) -> torch.Tensor:
    return torch.tensor(float_list, dtype=torch.float32).to(device)


class PPO(Learner):

    def __init__(self, agent: Core, device):
        self.agent = agent
        self.device = device

        self.lr = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_coef = 0.2
        self.norm_adv = True
        self.clip_vloss = True
        self.ent_coef = 0.0
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.target_kl = None

        self.update_epochs = 10
        self.num_minibatches = 32

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr, eps=1e-5)


    def reset(self, time = 0.0):
        frac = 1.0 - time
        lrnow = frac * self.lr
        self.optimizer.param_groups[0]["lr"] = lrnow


    def learn(self, obs: Context_Collector, actions: List[Any], logprobs: List[Any], rewards: List[List[float]], values: List[Any], next_dones: List[List[bool]], last_value: Any, last_done: List[bool]):
        """
        obs: Context_Collector
        actions: list of tensor of shape (batch_size, action_size)
        logprobs: list of tensor of shape (batch_size)
        rewards: list of floats of length batch_size
        values: list tensor of shape (batch_size)
        next_dones: list of bools of length batch_size
        last_value: tensor of shape (batch_size)
        last_done: list of bools of length batch_size
        """
        # Use dim 0 as context length dimension
        # obs = torch.stack(obs, dim=0)
        actions = torch.stack(actions, dim=0)
        logprobs = torch.stack(logprobs, dim=0)
        rewards = torch.stack([convert_list_of_float_to_float_tensor(r, self.device) for r in rewards], dim=0)
        values = torch.stack(values, dim=0)

        sequence_size = actions.shape[0]
        batch_size = actions.shape[1]
        minibatch_size = max(sequence_size // self.num_minibatches, 8)

        with torch.no_grad():
            advantages = []
            lastgaelam = torch.zeros(batch_size).to(self.device)
            for t in reversed(range(sequence_size)):
                if t == sequence_size - 1:
                    nextnonterminal = 1.0 - convert_list_of_bool_to_float_tensor(last_done, self.device)
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - convert_list_of_bool_to_float_tensor(next_dones[t], self.device)
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                advantages.append(lastgaelam)
            advantages.reverse()
            advantages = torch.stack(advantages, dim=0)
            returns = advantages + values

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(self.update_epochs):
            # np.random.shuffle(b_inds)
            for start in range(0, sequence_size, minibatch_size):
                end = start + minibatch_size

                _, b_newlogprob, b_entropy, b_newvalue = self.agent.get_action_and_value(
                    obs[start:end].make_batch(batch_led=True),
                    torch.transpose(actions[start:end, ...], 0, 1)
                    )
                
                mb_log_prob = torch.transpose(logprobs[start:end, ...], 0, 1)
                mb_value = torch.transpose(values[start:end, ...], 0, 1)
                mb_advantages = torch.transpose(advantages[start:end, ...], 0, 1)
                mb_returns = torch.transpose(returns[start:end, ...], 0, 1)
                
                logratio = b_newlogprob - mb_log_prob
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                if self.norm_adv and torch.numel(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if self.clip_vloss:
                    v_loss_unclipped = (b_newvalue - mb_returns) ** 2
                    v_clipped = mb_value + torch.clamp(
                        b_newvalue - mb_value,
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((b_newvalue - mb_returns) ** 2).mean()

                entropy_loss = b_entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break


