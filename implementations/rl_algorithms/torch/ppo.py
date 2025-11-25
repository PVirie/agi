from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from interfaces.learning import Learner
from interfaces.agent import Agent_Core


class PPO(Learner):

    def __init__(self, agent: Agent_Core, device):
        # agent = Agent(envs).to(device)
        self.parameters = agent.parameters()
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

        self.optimizer = optim.Adam(self.parameters, lr=self.lr, eps=1e-5)


    def reset(self, time = 0.0):
        frac = 1.0 - time
        lrnow = frac * self.lr
        self.optimizer.param_groups[0]["lr"] = lrnow
        
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.next_dones = []
        self.values = []


    def collect(self, obs, value, action, logprob, reward, termination, truncation):
        self.obs.append(obs)
        self.values.append(value)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)

        next_done = np.logical_or(termination, truncation)
        self.next_dones.append(next_done)


    def learn(self, last_value, last_termination, last_truncation):
        # bootstrap value if not done
        obs = torch.stack(self.obs)
        actions = torch.stack(self.actions)
        logprobs = torch.stack(self.logprobs)
        rewards = torch.stack(self.rewards)
        next_dones = torch.stack(self.next_dones)
        values = torch.stack(self.values)

        last_done = np.logical_or(last_termination, last_truncation)

        with torch.no_grad():
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(rewards.size(0))):
                if t == rewards.size(0) - 1:
                    nextnonterminal = 1.0 - last_done
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - next_dones[t]
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        batch_size = b_returns.shape[0]
        minibatch_size = batch_size // self.num_minibatches

        # flatten the batch
        b_obs = obs.reshape((batch_size, -1))
        b_logprobs = logprobs.reshape(batch_size)
        b_actions = actions.reshape((batch_size, -1))
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)
        b_values = values.reshape(batch_size)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break


