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
                 lr=3e-4, minibatch_size=8,
                 gamma=0.99,
                 gamma_intrinsic=0.99,
                 gae_lambda=None,
                 ext_coef=2.0,
                 int_coef=1.0,
                 use_graph=False):
        """
        PPO with dual value estimation (extrinsic + intrinsic/exploration), e.g. for RND.

        The policy model's get_log_probability_with_value must return values of shape
        (batch_size, context_length + 1, 2) where [..., 0] is the extrinsic value and
        [..., 1] is the intrinsic (exploration) value.

        Each stream gets its own GAE (own discount), its own return target and its own value loss.
        The policy advantage is ext_coef * adv_ext + int_coef * adv_int (Burda et al. 2018).
        Both streams are episodic: they reset on next_dones.

        gamma: discount of the extrinsic stream.
        gamma_intrinsic: discount of the intrinsic stream.
        use_graph: if True, `aux` passed to learn() is interpreted as causal links and used for
                   graph-based credit assignment on both streams independently; otherwise `aux`
                   is ignored and standard GAE is used.

        Issue: NaN when the sample actions are not on-policy.
        """
        self.policy_model = policy_model
        self.device = device

        self.lr = lr
        self.gamma = 0.99 if gamma is None else gamma
        self.gamma_intrinsic = 0.99 if gamma_intrinsic is None else gamma_intrinsic
        self.gae_lambda = 0.95 if gae_lambda is None else gae_lambda
        self.ext_coef = ext_coef
        self.int_coef = int_coef
        self.clip_coef = 0.2
        self.norm_adv = True
        self.clip_vloss = True
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.target_kl = None

        self.update_epochs = 4
        self.minibatch_size = minibatch_size

        self.use_graph = use_graph
        # How causal credit received by a step is aggregated across the future steps it caused.
        # 'amax' keeps the best future return; 'mean' averages them.
        self.causal_reduce = 'mean'  # one of {'amax', 'mean'}
        # Weight of the causal credit against the step's own GAE return. Fixed, so a step that is
        # referenced by many descendants does not lose its own learning signal to the average.
        self.causal_alpha = 0.8
        # Discount applied once per causal hop; kept separate from gamma so the graph shortcut
        # can be priced independently of the temporal discount.
        self.causal_decay = 0.99

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
              aux: Any = None
            ):
        """
        obs: np array of shape (batch_size, context_length + 1, ...)
        last_actions: np array of shape (batch_size, context_length + 1, ...)
        rewards: list (context_length) of np array of shape (batch_size, 2); [:, 0] extrinsic, [:, 1] intrinsic
        next_dones: list (context_length) of bools of length batch_size
        valid_actions: np array of shape (batch_size, context_length, ...)
        masks: np array of shape (batch_size, context_length)
        aux: np array of shape (batch_size, context_length + 1, cause_size); only used when use_graph is True

        Note that obs and last_actions include the context length + 1 items.
        This corresponds to the observations after taking the last actions.

        In graph mode, aux (causes) is a list of previous context indices that cause the current one.
        All return values from the current context index should be propagated to the cause context indices as well, not just the previous step.
        """
        # Use dim 0 as context length dimension
        b_obs = convert_np_array_to_float_tensor(obs, self.device)
        b_actions = convert_np_array_to_float_tensor(last_actions, self.device)
        b_rewards = convert_list_of_np_array_to_float_tensor(rewards, self.device)   # (batch_size, context_length, 2)
        b_next_dones = convert_list_of_list_of_bool_to_float_tensor(next_dones, self.device)
        b_valid_actions = convert_np_array_to_bool_tensor(valid_actions, self.device) if valid_actions is not None else None
        b_causes = convert_list_of_np_array_to_int_tensor(aux, self.device) if (self.use_graph and aux is not None) else None

        if b_rewards.dim() != 3 or b_rewards.shape[2] != 2:
            raise ValueError(f"rewards must be a list of (batch_size, 2) arrays [extrinsic, intrinsic]; got stacked shape {tuple(b_rewards.shape)}")

        batch_size = b_rewards.shape[0]
        sequence_size = b_rewards.shape[1]
        num_streams = b_rewards.shape[2]

        b_masks = torch.ones((batch_size, sequence_size), device=self.device) if masks is None else convert_np_array_to_float_tensor(masks, self.device)
        gammas = torch.tensor([self.gamma, self.gamma_intrinsic], dtype=torch.float32, device=self.device)   # (2,)
        stream_coefs = torch.tensor([self.ext_coef, self.int_coef], dtype=torch.float32, device=self.device)  # (2,)

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
            values_with_last = torch.cat(values_with_last_list, dim=0)   # (batch_size, context_length + 1, 2)

            if values_with_last.dim() != 3 or values_with_last.shape[2] != num_streams:
                raise ValueError(f"policy model must return values of shape (batch_size, context_length + 1, 2); got {tuple(values_with_last.shape)}")

            values = values_with_last[:, :-1, :]  # (batch_size, context_length, 2)

            # Same backward pass as ppo_graph, vectorised over the trailing stream dimension: each
            # stream runs its own GAE with its own discount, and causal credit is propagated per
            # stream independently (buffers carry the stream dimension).
            if b_causes is not None:
                use_mean = (self.causal_reduce == 'mean')
                if use_mean:
                    causal_sum = torch.zeros((batch_size, sequence_size, num_streams), device=self.device)
                    causal_count = torch.zeros((batch_size, sequence_size, num_streams), device=self.device)
                    causal_returns = None
                else:
                    causal_returns = torch.full((batch_size, sequence_size, num_streams), float('-inf'), device=self.device)
                    causal_sum = causal_count = None
            else:
                use_mean = False
                causal_returns = causal_sum = causal_count = None
            advantages = []
            returns_list = []
            lastgaelam = torch.zeros((batch_size, num_streams), device=self.device)
            for t in reversed(range(sequence_size)):
                nextnonterminal = (1.0 - b_next_dones[:, t])[:, None]          # (batch_size, 1)
                nextvalues = values_with_last[:, t + 1]                         # (batch_size, 2)
                delta = b_rewards[:, t] + gammas * nextvalues * nextnonterminal - values_with_last[:, t]
                lastgaelam = delta + gammas * self.gae_lambda * nextnonterminal * lastgaelam

                standard_return = lastgaelam + values[:, t]                     # (batch_size, 2)

                if b_causes is not None:
                    all_causes = b_causes[:, t, :]                              # (batch_size, cause_size)
                    valid_mask = (all_causes >= 0) & (all_causes < t) & b_masks[:, t].bool()[:, None]
                    valid_mask = valid_mask[:, :, None].expand(-1, -1, num_streams)                 # (batch_size, cause_size, 2)
                    safe_causes = all_causes.clamp(0, sequence_size - 1)        # prevent OOB on scatter
                    safe_causes = safe_causes[:, :, None].expand(-1, -1, num_streams)               # (batch_size, cause_size, 2)

                    if use_mean:
                        count_t = causal_count[:, t]                            # (batch_size, 2)
                        mean_credit = causal_sum[:, t] / count_t.clamp(min=1)
                        has_signal = count_t > 0
                        effective_return = torch.where(
                            has_signal,
                            (1.0 - self.causal_alpha) * standard_return + self.causal_alpha * mean_credit,
                            standard_return,
                        )

                        src = effective_return[:, None, :].expand(-1, b_causes.shape[2], -1)       # (batch_size, cause_size, 2)
                        src_val = (src * self.causal_decay).masked_fill(~valid_mask, 0.0)
                        ones = valid_mask.float()
                        causal_sum = causal_sum.scatter_add(1, safe_causes, src_val)
                        causal_count = causal_count.scatter_add(1, safe_causes, ones)
                    else:
                        effective_return = torch.max(standard_return, causal_returns[:, t])

                        src = effective_return[:, None, :].expand(-1, b_causes.shape[2], -1)       # (batch_size, cause_size, 2)
                        src = (src * self.causal_decay).masked_fill(~valid_mask, float('-inf'))
                        causal_returns = causal_returns.scatter_reduce(1, safe_causes, src, reduce='amax', include_self=True)

                    advantages.append(effective_return - values[:, t])
                else:
                    # Plain GAE: use lastgaelam directly to avoid (x + v) - v rounding.
                    advantages.append(lastgaelam)

                # value targets stay the plain GAE returns; causal credit only shapes the policy gradient
                returns_list.append(standard_return)

            advantages.reverse()
            returns_list.reverse()
            advantages = torch.stack(advantages, dim=1)     # (batch_size, context_length, 2)
            returns = torch.stack(returns_list, dim=1)      # (batch_size, context_length, 2)

            # combine the per-stream advantages into the single policy advantage, then normalise
            advantages = (advantages * stream_coefs).sum(dim=-1)   # (batch_size, context_length)

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
                mb_newvalue = mb_newvalue[:, :-1, :]  # remove last value, (minibatch, context_length, 2)
                
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

                # Value loss: per-stream (clipped) squared error, summed over the two heads
                if self.clip_vloss:
                    v_loss_unclipped = (mb_newvalue - mb_returns) ** 2
                    v_clipped = mb_value + torch.clamp(
                        mb_newvalue - mb_value,
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * masked_mean(v_loss_max.sum(dim=-1), mb_masks)
                else:
                    v_loss = 0.5 * masked_mean(((mb_newvalue - mb_returns) ** 2).sum(dim=-1), mb_masks)

                entropy_loss = masked_mean(mb_entropy, mb_masks)

                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.all_parameters, self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        self.policy_model.eval()
