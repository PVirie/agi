from typing import Any, List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging

from interfaces.learning import Learner
from interfaces.core import Core, Context_Collector


def convert_list_of_bool_to_float_tensor(bool_list: List[bool], device) -> torch.Tensor:
    return torch.tensor([1.0 if b else 0.0 for b in bool_list], dtype=torch.float32).to(device)


def convert_np_array_to_float_tensor(np_array: np.ndarray, device) -> torch.Tensor:
    return torch.tensor(np_array, dtype=torch.float32).to(device)


def convert_list_of_np_array_to_float_tensor(np_array_list: List[np.ndarray], device) -> List[torch.Tensor]:
    before_transpose = [torch.tensor(arr, dtype=torch.float32).to(device) for arr in np_array_list]
    return torch.stack(before_transpose, dim=1)


def convert_list_of_float_to_float_tensor(float_list: List[float], device) -> torch.Tensor:
    before_transpose = torch.tensor(float_list, dtype=torch.float32).to(device)
    return torch.transpose(before_transpose, 0, 1)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False) -> torch.Tensor:
    masked_tensor = tensor * mask
    if dim is None:
        total_elements = mask.numel()
    else:
        total_elements = mask.sum(dim=dim, keepdim=keepdim)
    
    return masked_tensor.sum(dim=dim, keepdim=keepdim) / (total_elements + 1e-8)


def masked_std(tensor: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False) -> torch.Tensor:
    masked_tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim, keepdim=True)
    variance = masked_mean((masked_tensor - mean) ** 2, mask, dim=dim, keepdim=keepdim)
    return torch.sqrt(variance + 1e-8)


class PPO(Learner):

    def __init__(self, agent: Core, device, persistence_path=None):
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
                checkpoint = torch.load(f"{self.persistence_path}/ppo_checkpoint.pth", map_location=self.device)
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logging.info(f"Loaded PPO learner from {self.persistence_path}/ppo_checkpoint.pth")
            except FileNotFoundError:
                logging.info(f"No PPO learner checkpoint found at {self.persistence_path}/ppo_checkpoint.pth, starting fresh.")


    def save(self):
        if self.persistence_path is not None:
            torch.save({
                "optimizer_state_dict": self.optimizer.state_dict(),
            }, f"{self.persistence_path}/ppo_checkpoint.pth")
            logging.info(f"Saved PPO learner to {self.persistence_path}/ppo_checkpoint.pth")


    def learn(self, obs: Any, actions: Any, logprobs: List[Any], rewards: List[List[float]], values: List[Any], next_dones: List[List[bool]], last_value: Any, last_done: List[bool], masks: Any = None):
        """
        obs: np array of shape (batch_size, context_length, ...)
        actions: np array of shape (batch_size, context_length, ...)
        logprobs: list of np array of shape (batch_size)
        rewards: list of floats of length batch_size
        values: list np array of shape (batch_size)
        next_dones: list of bools of length batch_size
        last_value: np array of shape (batch_size)
        last_done: list of bools of length batch_size
        masks: np array of shape (batch_size, context_length)
        """
        # Use dim 0 as context length dimension
        obs = convert_np_array_to_float_tensor(obs, self.device)
        actions = convert_np_array_to_float_tensor(actions, self.device)
        logprobs = convert_list_of_np_array_to_float_tensor(logprobs, self.device)
        rewards = convert_list_of_float_to_float_tensor(rewards, self.device)
        values = convert_list_of_np_array_to_float_tensor(values, self.device)
        masks = torch.ones_like(rewards).to(self.device) if masks is None else convert_np_array_to_float_tensor(masks, self.device)

        batch_size = actions.shape[0]
        sequence_size = actions.shape[1]
        minibatch_size = max(sequence_size // self.num_minibatches, 8)

        with torch.no_grad():
            advantages = []
            lastgaelam = torch.zeros(batch_size).to(self.device)
            for t in reversed(range(sequence_size)):
                if t == sequence_size - 1:
                    nextnonterminal = 1.0 - convert_list_of_bool_to_float_tensor(last_done, self.device)
                    nextvalues = convert_np_array_to_float_tensor(last_value, self.device)
                else:
                    nextnonterminal = 1.0 - convert_list_of_bool_to_float_tensor(next_dones[t], self.device)
                    nextvalues = values[:, t + 1]
                delta = rewards[:, t] + self.gamma * nextvalues * nextnonterminal - values[:, t]
                lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                advantages.append(lastgaelam)
            advantages.reverse()
            advantages = torch.stack(advantages, dim=1)
            returns = advantages + values

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(self.update_epochs):
            # np.random.shuffle(b_inds)
            for start in range(0, sequence_size, minibatch_size):
                end = start + minibatch_size

                mb_log_prob = logprobs[:, start:end, ...]
                mb_value = values[:, start:end, ...]
                mb_advantages = advantages[:, start:end, ...] 
                mb_returns = returns[:, start:end, ...]
                mb_masks = masks[:, start:end, ...]

                if torch.sum(mb_masks) < 1e-8:
                    continue

                _, _, b_newlogprob, b_entropy, b_newvalue = self.agent.get_action_and_value(
                    obs, 
                    actions,
                    use_action=True,
                    use_grad=True
                )

                b_newlogprob = b_newlogprob[:, start:end, ...]
                b_entropy = b_entropy[:, start:end, ...]
                b_newvalue = b_newvalue[:, start:end, ...]
                
                logratio = b_newlogprob - mb_log_prob
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = masked_mean(-logratio, mb_masks)
                    approx_kl = masked_mean((ratio - 1) - logratio, mb_masks)
                    clipfracs += [masked_mean(((ratio - 1.0).abs() > self.clip_coef).float(), mb_masks).item()]

                if self.norm_adv and torch.numel(mb_advantages) > 1:
                    mb_adv_mean = masked_mean(mb_advantages, mb_masks)
                    mb_adv_std = masked_std(mb_advantages, mb_masks)
                    mb_advantages = (mb_advantages - mb_adv_mean) / (mb_adv_std + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = masked_mean(torch.max(pg_loss1, pg_loss2), mb_masks)

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
                    v_loss = 0.5 * masked_mean(v_loss_max, mb_masks)
                else:
                    v_loss = 0.5 * masked_mean((b_newvalue - mb_returns) ** 2, mb_masks)

                entropy_loss = masked_mean(b_entropy, mb_masks)
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break


