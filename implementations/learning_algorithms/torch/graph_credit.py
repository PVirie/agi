import torch


def compute_graph_gae(
        rewards: torch.Tensor,
        values_with_last: torch.Tensor,
        next_dones: torch.Tensor,
        gamma,
        gae_lambda: float,
        causes: torch.Tensor = None,
        masks: torch.Tensor = None,
        causal_reduce: str = 'mean',
        causal_alpha: float = 0.8,
        causal_decay: float = 0.99,
    ):
    """
    GAE with optional causal-graph credit assignment, computed independently for each reward stream.

    rewards: (batch_size, context_length, num_streams)
    values_with_last: (batch_size, context_length + 1, num_streams); the last entry is the bootstrap value
    next_dones: (batch_size, context_length)
    gamma: float, or a tensor of shape (num_streams,) for a per-stream discount
    causes: (batch_size, context_length + 1, cause_size) int tensor of past context indices that caused each step
            (-1 = empty slot), or None for plain GAE
    masks: (batch_size, context_length); steps with mask 0 do not propagate credit. None means all valid.
    causal_reduce: how a cause aggregates the credit of the future steps it caused;
                   'amax' keeps the best one, 'mean' blends its own return with their average at ratio causal_alpha.
    causal_decay: discount applied once per causal hop.

    Returns (advantages, returns), both (batch_size, context_length, num_streams).
    The returns are the plain GAE returns (value targets); causal credit only shapes the advantages.
    """
    batch_size, sequence_size, num_streams = rewards.shape
    device = rewards.device
    values = values_with_last[:, :-1]

    # Single backward pass. For every step t we propagate its effective return back to all
    # past causes c (c < t). Because causes always point backward, by the time the loop
    # reaches c every future t > c has already deposited its credit, so c's aggregated
    # causal credit is fully populated when it is consumed.
    # For 'amax' we keep a single running buffer initialised to -inf (the max identity).
    # For 'mean' we keep running sum/count buffers and divide on read, so that invalid
    # cause slots can be excluded from the average without corrupting it.
    if causes is not None:
        use_mean = (causal_reduce == 'mean')
        if use_mean:
            causal_sum = torch.zeros((batch_size, sequence_size, num_streams), device=device)
            causal_count = torch.zeros((batch_size, sequence_size, num_streams), device=device)
        else:
            causal_returns = torch.full((batch_size, sequence_size, num_streams), float('-inf'), device=device)
        cause_size = causes.shape[2]

    advantages = []
    returns_list = []
    lastgaelam = torch.zeros((batch_size, num_streams), device=device)
    for t in reversed(range(sequence_size)):
        nextnonterminal = (1.0 - next_dones[:, t])[:, None]                # (batch_size, 1)
        nextvalues = values_with_last[:, t + 1]                             # (batch_size, num_streams)
        delta = rewards[:, t] + gamma * nextvalues * nextnonterminal - values_with_last[:, t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

        standard_return = lastgaelam + values[:, t]                         # (batch_size, num_streams)

        if causes is not None:
            # Blend the standard return with any causal credit received from future steps, then
            # propagate that effective return back to all past causes of t.
            # Propagating the effective (post-aggregation) value means transitive chains
            # (c caused t, t caused t') are handled without an extra pass.
            all_causes = causes[:, t, :]                                    # (batch_size, cause_size)
            valid_mask = (all_causes >= 0) & (all_causes < t)
            if masks is not None:
                valid_mask = valid_mask & masks[:, t].bool()[:, None]
            valid_mask = valid_mask[:, :, None].expand(-1, -1, num_streams)             # (batch_size, cause_size, num_streams)
            safe_causes = all_causes.clamp(0, sequence_size - 1)            # prevent OOB on scatter
            safe_causes = safe_causes[:, :, None].expand(-1, -1, num_streams)           # (batch_size, cause_size, num_streams)

            if use_mean:
                # fall back to the standard return when no future step deposited any credit
                count_t = causal_count[:, t]
                mean_credit = causal_sum[:, t] / count_t.clamp(min=1)
                has_signal = count_t > 0
                effective_return = torch.where(
                    has_signal,
                    (1.0 - causal_alpha) * standard_return + causal_alpha * mean_credit,
                    standard_return,
                )

                # vectorised scatter-add of sum and count; invalid slots add 0 / 0
                src = effective_return[:, None, :].expand(-1, cause_size, -1)           # (batch_size, cause_size, num_streams)
                src_val = (src * causal_decay).masked_fill(~valid_mask, 0.0)
                causal_sum = causal_sum.scatter_add(1, safe_causes, src_val)
                causal_count = causal_count.scatter_add(1, safe_causes, valid_mask.float())
            else:
                effective_return = torch.max(standard_return, causal_returns[:, t])

                # vectorised scatter-max; invalid slots get -inf, a no-op under max
                src = effective_return[:, None, :].expand(-1, cause_size, -1)           # (batch_size, cause_size, num_streams)
                src = (src * causal_decay).masked_fill(~valid_mask, float('-inf'))
                causal_returns = causal_returns.scatter_reduce(1, safe_causes, src, reduce='amax', include_self=True)

            advantages.append(effective_return - values[:, t])
        else:
            # Plain GAE: use lastgaelam directly to avoid (x + v) - v rounding.
            advantages.append(lastgaelam)

        returns_list.append(standard_return)

    advantages.reverse()
    returns_list.reverse()
    return torch.stack(advantages, dim=1), torch.stack(returns_list, dim=1)
