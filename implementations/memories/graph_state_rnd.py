import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utilities.safe_torch_module import Safe_nn_Module
from utilities.atari.rnd import Running_Mean_Std, RND_Network, mlp_trunk


class Graph_State_Tokenizer:
    """Turns the local view of an NP_Graph_Memory (what the policy reads through get_node_context) into categorical tokens.

    Fields, per env:
    - content: the head node and its C - 1 edge slots, node_dim values each; each value is a token of the vocabulary,
      and a free edge slot gets its own "empty" value (get_node_context zeroes free slots, which would alias token 0).
    - degree (include_degree): the number of used edges of the head and of every neighbour in slot order, 0..max_edges.
      Two views with the same content but a different wiring are different graph states.
    Every field owns its own embedding rows (token = field offset + value), so the same value seen at a different slot
    reaches the predictor as a different input, just as the policy sees slots in order.
    """

    def __init__(self, vocab_size, max_edges_per_node, node_dim=1, include_degree=True):
        self.vocab_size = vocab_size
        self.max_edges = max_edges_per_node
        self.C = 1 + max_edges_per_node
        self.node_dim = node_dim
        self.include_degree = include_degree

        self.num_content_fields = self.C * node_dim
        self.content_values = vocab_size + 1  # + the empty-slot value
        self.empty_value = vocab_size
        self.num_degree_fields = self.C if include_degree else 0
        self.degree_values = max_edges_per_node + 1

        self.content_offsets = np.arange(self.num_content_fields, dtype=np.int64) * self.content_values
        degree_base = self.num_content_fields * self.content_values
        self.degree_offsets = degree_base + np.arange(self.num_degree_fields, dtype=np.int64) * self.degree_values
        self.num_tokens = degree_base + self.num_degree_fields * self.degree_values
        self.num_fields = self.num_content_fields + self.num_degree_fields


    def __call__(self, graph_memory, batch_indices):
        """graph_memory: NP_Graph_Memory; batch_indices: (N,) -> int64 tokens (N, num_fields)"""
        batch_indices = np.asarray(batch_indices, dtype=np.int64)
        if graph_memory.edges.shape[2] != self.max_edges or graph_memory.nodes.shape[2] != self.node_dim:
            raise ValueError(
                f"Graph memory shape (max_edges={graph_memory.edges.shape[2]}, node_dim={graph_memory.nodes.shape[2]}) "
                f"does not match the tokenizer (max_edges={self.max_edges}, node_dim={self.node_dim})"
            )
        heads = graph_memory.head[batch_indices]                                        # (N,)
        neighbors = graph_memory.edges[batch_indices, heads, :]                         # (N, max_edges)
        used = graph_memory.next_free_edge[batch_indices, heads]                        # (N,)
        valid = np.arange(self.max_edges)[None, :] < used[:, None]                      # (N, max_edges)

        view_nodes = np.concatenate([heads[:, None], neighbors], axis=1)                # (N, C)
        view_valid = np.concatenate([np.ones_like(valid[:, :1]), valid], axis=1)        # (N, C)

        values = graph_memory.nodes[batch_indices[:, None], view_nodes, :]              # (N, C, node_dim)
        values = np.rint(values).astype(np.int64)
        if values.size > 0 and (values.min() < 0 or values.max() >= self.vocab_size):
            logging.warning(f"Graph node values outside [0, {self.vocab_size}) are clipped for graph-state RND.")
            values = np.clip(values, 0, self.vocab_size - 1)
        values = np.where(view_valid[:, :, None], values, self.empty_value)
        tokens = [values.reshape(len(batch_indices), -1) + self.content_offsets[None, :]]

        if self.include_degree:
            degrees = graph_memory.next_free_edge[batch_indices[:, None], view_nodes].astype(np.int64)
            degrees = np.where(view_valid, degrees, 0)
            tokens.append(degrees + self.degree_offsets[None, :])

        return np.concatenate(tokens, axis=1)


class Episodic_State_Counter:
    """Per env, visit counts of graph states (token tuples) since the env's graph was last reset:
    r_episodic = 1 / sqrt(count). Also keeps every state ever reached, as an exploration-progress diagnostic."""

    def __init__(self, num_envs):
        self.counts = [dict() for _ in range(num_envs)]
        self.states_seen = set()

    def visit(self, tokens, env_indices):
        rewards = np.empty((len(env_indices),), dtype=np.float64)
        for k, i in enumerate(env_indices):
            key = tokens[k].tobytes()
            n = self.counts[i].get(key, 0) + 1
            self.counts[i][key] = n
            rewards[k] = n ** -0.5
            self.states_seen.add(key)
        return rewards

    def reset(self, env_index):
        self.counts[env_index].clear()


class Graph_State_RND(nn.Module, Safe_nn_Module):
    """Random Network Distillation over the local state of the graph memory, an exploration bonus for internal actions.

    Mirrors utilities.atari.rnd.RND_Model, with the RAM replaced by the graph view of Graph_State_Tokenizer
    (head + neighbour contents, and optionally their degrees), all embedded as categorical tokens.
    - Target: frozen random embeddings + MLP -> linear(rep_size).
      Predictor: its own trained embeddings + same MLP -> 2 hidden ReLU layers -> linear(rep_size).
    - r_intrinsic = mean_k (predictor(g)_k - target(g)_k)^2 / running_std(per-env discounted intrinsic return)
    - The predictor trains on a random `update_proportion` of each batch.
    - episodic=True (NGU): r_intrinsic = r_episodic * clip(1 + (error - mean(error)) / std(error), 1, max_modulation),
      with r_episodic = 1 / sqrt(visits of the graph state since the env's graph was reset).
      Call reset_episode(i) whenever env i's graph memory is reset.
    - stats_window_steps: if set, the reward (and error) statistics forget like a moving average over about that many steps.

    Usage: after graph_memory.execute(...), call compute_intrinsic_rewards(graph_memory, batch_indices) with the envs
    whose internal action should be rewarded; the reward describes the resulting graph state.
    """

    def __init__(
        self,
        num_envs: int,
        vocab_size: int,
        max_edges_per_node: int,
        node_dim: int = 1,
        include_degree: bool = True,
        embedding_size: int = 16,
        trunk_hidden_size: int = 256,
        rep_size: int = 256,
        predictor_hidden_size: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        update_proportion: float = None,
        warmup_steps: int = 128,
        episodic: bool = False,
        max_modulation: float = 5.0,
        stats_window_steps: int = None,
        device="cpu",
        persistence_path=None,
        seed: int = 0,
    ):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="graph_state_rnd_model", device=device, persistence_path=persistence_path)
        self.num_envs = num_envs
        self.tokenizer = tokenizer = Graph_State_Tokenizer(vocab_size, max_edges_per_node, node_dim, include_degree)
        self.obs_size = obs_size = tokenizer.num_fields * embedding_size
        self.gamma = gamma
        self.update_proportion = min(1.0, 32.0 / num_envs) if update_proportion is None else update_proportion
        # rewards stay 0 until the statistics have seen this many states (the first states are all near-empty graphs)
        self.warmup_states = warmup_steps * num_envs
        if stats_window_steps is not None and stats_window_steps < warmup_steps:
            raise ValueError(f"stats_window_steps ({stats_window_steps}) must be at least warmup_steps ({warmup_steps})")
        self.episodic = episodic
        self.max_modulation = max_modulation
        self.state_counter = Episodic_State_Counter(num_envs)
        self.diagnostic_sums = {}
        self.diagnostic_count = 0
        # private RNGs so enabling the bonus does not shift the global streams used by the policy and the envs
        self.rng = np.random.default_rng(seed)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.target_net = RND_Network(tokenizer.num_tokens, embedding_size, nn.Sequential(
                mlp_trunk(obs_size, trunk_hidden_size),
                nn.Linear(trunk_hidden_size, rep_size),
            ))
            self.predictor_net = RND_Network(tokenizer.num_tokens, embedding_size, nn.Sequential(
                mlp_trunk(obs_size, trunk_hidden_size),
                nn.Linear(trunk_hidden_size, predictor_hidden_size), nn.ReLU(),
                nn.Linear(predictor_hidden_size, predictor_hidden_size), nn.ReLU(),
                nn.Linear(predictor_hidden_size, rep_size),
            ))
            for m in list(self.target_net.modules()) + list(self.predictor_net.modules()):
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.zeros_(m.bias)
        for p in self.target_net.parameters():
            p.requires_grad = False

        stats_window = None if stats_window_steps is None else stats_window_steps * num_envs
        self.reward_stats = Running_Mean_Std(window=stats_window)
        if episodic:
            # registered only in episodic mode, so non-episodic checkpoints keep loading
            self.error_stats = Running_Mean_Std(window=stats_window)
        # discounted intrinsic return per env, never reset (the RND return is non-episodic)
        self.register_buffer("reward_filter", torch.zeros((num_envs,), dtype=torch.float64), persistent=False)
        self.to(device)

        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=lr)
        # separate saver: Safe_nn_Module.load stops after the first module of a multi-module dict
        self.learner = Safe_nn_Module(
            name="graph_state_rnd_learner", device=device, persistence_path=persistence_path,
            modules={"graph_state_rnd_learner": self.optimizer}
        )

        self.load()
        # Adam's state_dict carries the lr it was saved with; the configured lr wins on resume
        for group in self.optimizer.param_groups:
            group["lr"] = lr


    def compute_intrinsic_rewards(self, graph_memory, batch_indices=None):
        """
        graph_memory: NP_Graph_Memory, after the internal action has been executed.
        batch_indices: the envs to reward (default: all). Rewards are computed for their current graph state.
        Returns np float32 (N,) intrinsic rewards and trains the predictor one step.
        """
        if batch_indices is None:
            batch_indices = np.arange(self.num_envs)
        batch_indices = np.asarray(batch_indices, dtype=np.int64)
        if len(batch_indices) == 0:
            return np.zeros((0,), dtype=np.float32)

        tokens_np = self.tokenizer(graph_memory, batch_indices)
        tokens = torch.as_tensor(tokens_np, device=self.device)
        no_continuous = torch.zeros((len(batch_indices), 0), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            target_out = self.target_net(tokens, no_continuous)
            raw_r_i = (self.predictor_net(tokens, no_continuous) - target_out).pow(2).mean(dim=1)

        keep = np.nonzero(self.rng.random(len(batch_indices)) < self.update_proportion)[0]
        if len(keep) > 0:
            keep = torch.as_tensor(keep, dtype=torch.long, device=self.device)
            loss = (self.predictor_net(tokens[keep], no_continuous[keep]) - target_out[keep]).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        raw_r_i = raw_r_i.to(torch.float64)
        idx = torch.as_tensor(batch_indices, dtype=torch.long, device=self.device)
        self.reward_filter[idx] = self.reward_filter[idx] * self.gamma + raw_r_i
        self.reward_stats.update(self.reward_filter[idx])
        warmed_up = self.reward_stats.count.item() >= self.warmup_states
        if warmed_up:
            rnd_reward = raw_r_i / torch.sqrt(self.reward_stats.var + 1e-8)
        else:
            rnd_reward = torch.zeros_like(raw_r_i)

        episodic_reward = torch.as_tensor(self.state_counter.visit(tokens_np, batch_indices), device=self.device)

        modulator = None
        if self.episodic:
            self.error_stats.update(raw_r_i)
            # the episodic reward is bounded, so it is paid during the warmup too, unmodulated
            modulator = torch.ones_like(raw_r_i)
            if warmed_up:
                z = (raw_r_i - self.error_stats.mean) / torch.sqrt(self.error_stats.var + 1e-8)
                modulator = (1.0 + z).clamp(1.0, self.max_modulation)
            reward = episodic_reward * modulator
        else:
            reward = rnd_reward

        self._accumulate_diagnostics(
            len(batch_indices),
            graph_rnd_error=raw_r_i, graph_rnd_reward=rnd_reward, graph_episodic_reward=episodic_reward,
            graph_modulator=modulator, graph_intrinsic_reward=reward,
        )
        return reward.cpu().numpy().astype(np.float32)


    def reset_episode(self, env_index):
        """Call when env_index's graph memory is reset: clears its episodic counts. The RND state itself is non-episodic."""
        self.state_counter.reset(env_index)


    def _accumulate_diagnostics(self, n, **tensors):
        # summed on the device; synced only in pop_diagnostics
        for key, t in tensors.items():
            if t is not None:
                self.diagnostic_sums[key] = self.diagnostic_sums.get(key, 0.0) + t.sum()
        self.diagnostic_count += n


    def pop_diagnostics(self):
        """Per-state means since the last call and the number of distinct graph states reached since start; resets the means."""
        n = max(self.diagnostic_count, 1)
        out = {key: float(v) / n for key, v in self.diagnostic_sums.items()}
        out["graph_states_seen"] = len(self.state_counter.states_seen)
        self.diagnostic_sums = {}
        self.diagnostic_count = 0
        return out


    def save(self, num_to_keep=2, override_persistence_path=None):
        Safe_nn_Module.save(self, num_to_keep=num_to_keep, override_persistence_path=override_persistence_path)
        self.learner.save(num_to_keep=num_to_keep, override_persistence_path=override_persistence_path)


    def load(self, override_persistence_path=None):
        Safe_nn_Module.load(self, override_persistence_path=override_persistence_path)
        self.learner.load(override_persistence_path=override_persistence_path)
        # Adam's load_state_dict does not check shapes; discard a mismatching optimizer state
        for p in self.predictor_net.parameters():
            if any(torch.is_tensor(v) and v.dim() > 0 and v.shape != p.shape for v in self.optimizer.state.get(p, {}).values()):
                logging.warning("Graph-state RND optimizer checkpoint does not match the predictor shapes, discarding it.")
                self.optimizer.state.clear()
                break



if __name__ == "__main__":
    from implementations.memories.graph_memory import NP_Graph_Memory

    def check(name, cond):
        if not cond:
            raise AssertionError(f"FAIL [{name}]")
        print(f"  PASS  {name}")

    V, E = 8, 3
    print("TEST 1: tokenizer marks free slots as empty, not token 0")
    m = NP_Graph_Memory(num_batches=2, num_nodes=6, max_edges_per_node=E, node_dim=1, start_node_value=V - 1)
    m.create(np.array([0]), np.array([[0.0]]))  # batch 0: head <-> node 1 (value 0)
    tok = Graph_State_Tokenizer(V, E, node_dim=1, include_degree=True)
    t = tok(m, np.array([0, 1]))
    check("token shape", t.shape == (2, 2 * (1 + E)))
    check("tokens in range", t.min() >= 0 and t.max() < tok.num_tokens)
    content = t[:, :1 + E] - tok.content_offsets[None, :]
    check("batch 0 head value", content[0, 0] == V - 1)
    check("batch 0 slot 0 value 0", content[0, 1] == 0)
    check("batch 0 free slots empty", np.all(content[0, 2:] == tok.empty_value))
    check("batch 1 all slots empty", np.all(content[1, 1:] == tok.empty_value))
    degree = t[:, 1 + E:] - tok.degree_offsets[None, :]
    check("batch 0 degrees", list(degree[0]) == [1, 1, 0, 0])
    check("batch 1 degrees", list(degree[1]) == [0, 0, 0, 0])

    print("TEST 2: rewards, training and episodic counts")
    rnd = Graph_State_RND(num_envs=2, vocab_size=V, max_edges_per_node=E, warmup_steps=2, episodic=True, update_proportion=1.0)
    r1 = rnd.compute_intrinsic_rewards(m)
    check("reward shape", r1.shape == (2,))
    r2 = rnd.compute_intrinsic_rewards(m, np.array([1]))
    check("subset reward shape", r2.shape == (1,))
    check("repeat visit decays episodic reward", r2[0] < r1[1])
    rnd.reset_episode(1)
    r3 = rnd.compute_intrinsic_rewards(m, np.array([1]))
    check("reset restores episodic reward", r3[0] >= r2[0])

    errors = []
    for _ in range(200):
        rnd.compute_intrinsic_rewards(m)
        errors.append(rnd.pop_diagnostics()["graph_rnd_error"])
    check("predictor learns a repeated state", errors[-1] < errors[0])
    check("empty batch", rnd.compute_intrinsic_rewards(m, np.array([], dtype=np.int64)).shape == (0,))
    print("All tests passed.")
