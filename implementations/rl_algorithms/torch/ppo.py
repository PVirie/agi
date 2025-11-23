from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time


class PPO:

    def __init__(self, agent, observation_dim: int, action_dim: int, device):
        # agent = Agent(envs).to(device)
        self.parameters = agent.parameters()
        self.agent = agent
        self.device = device

        self.observation_dim = observation_dim
        self.action_dim = action_dim
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

        # ALGO Logic: Storage setup
        # obs = torch.zeros((args.num_steps, observation_dim)).to(device)
        # actions = torch.zeros((args.num_steps, action_dim)).to(device)
        # logprobs = torch.zeros((args.num_steps)).to(device)
        # rewards = torch.zeros((args.num_steps)).to(device)
        # dones = torch.zeros((args.num_steps)).to(device)
        # values = torch.zeros((args.num_steps)).to(device)

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


    def collect(self, obs, value, action, logprob, reward, terminations, truncations):
        self.obs.append(obs)
        self.values.append(value)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)

        next_done = np.logical_or(terminations, truncations)
        self.next_dones.append(next_done)


    def learn(self, last_value, last_terminations, last_truncations):
        # bootstrap value if not done
        obs = torch.stack(self.obs)
        actions = torch.stack(self.actions)
        logprobs = torch.stack(self.logprobs)
        rewards = torch.stack(self.rewards)
        next_dones = torch.stack(self.next_dones)
        values = torch.stack(self.values)

        last_done = np.logical_or(last_terminations, last_truncations)

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

        # flatten the batch
        b_obs = obs.reshape((-1, self.observation_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, self.action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        batch_size = b_obs.shape[0]
        minibatch_size = batch_size // self.num_minibatches
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


