import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utilities.safe_torch_module import Safe_nn_Module


# RAM bytes RND reads per game, keyed by the game name in the id (e.g. "ALE/MontezumaRevenge-v5" -> "MontezumaRevenge"),
# split into "categorical" bytes (ids: rooms, levels, items; embedded) and "continuous" bytes (magnitudes: positions; normalized).
# The other bytes (timers, enemy positions, ...) change on their own and would keep every state novel.
# Games not listed here give RND the full RAM as continuous bytes.
# https://github.com/mila-iqia/atari-representation-learning/blob/master/atariari/benchmark/ram_annotations.py
RND_RAM_MASKS = {
    "MontezumaRevenge": {
        # Verified AtariARI: room_number(3), level(57), items_count(61), room_state(62)
        "categorical": [3, 57, 61, 62],
        # Verified AtariARI: player_x(42), player_y(43)
        "continuous": [42, 43],
    },
    "Pitfall": {
        # Screen counter from 6502 source ($83 - $80 = 3).
        # (If sticking strictly to AtariARI, leave categorical empty: [])
        "categorical": [],
        # Verified AtariARI: player_x(97), player_y(105)
        "continuous": [97, 105],
    },
    "Adventure": {
        # Verified 6502 asm ($8A, $9D, $A0, $A1, $A2):
        # Room(10), Carried Object(29), Yellow/White/Black Castle Gates(32, 33, 34)
        "categorical": [10, 29, 32, 33, 34],
        # Verified 6502 asm ($8B, $8C): Ball/Player X(11), Ball/Player Y(12)
        "continuous": [11, 12],
    },
}


def game_name(game_id):
    """ "ALE/MontezumaRevenge-v5" or "MontezumaRevengeNoFrameskip-v4" -> "MontezumaRevenge" """
    name = game_id.split("/")[-1].split("-")[0]
    for suffix in ("NoFrameskip", "Deterministic"):
        name = name.removesuffix(suffix)
    return name


RAM_VALUES = 256  # a RAM byte is uint8


def parse_ram_mask(mask, ram_size, name):
    """One game's RND_RAM_MASKS entry -> (categorical bytes, continuous bytes).
    None (unlisted game) reads the full RAM as continuous; a flat list of bytes is read as continuous."""
    if mask is None:
        categorical, continuous = [], list(range(ram_size))
    elif isinstance(mask, dict):
        unknown = set(mask) - {"categorical", "continuous"}
        if unknown:
            raise ValueError(f"Invalid RND RAM mask for {name}: unknown keys {sorted(unknown)}")
        categorical, continuous = mask.get("categorical", []), mask.get("continuous", [])
    else:
        categorical, continuous = [], mask

    categorical = np.asarray(categorical, dtype=np.int64).reshape(-1)
    continuous = np.asarray(continuous, dtype=np.int64).reshape(-1)
    both = np.concatenate([categorical, continuous])
    if len(both) == 0 or both.min() < 0 or both.max() >= ram_size or len(np.unique(both)) != len(both):
        raise ValueError(f"Invalid RND RAM mask for {name}: categorical {categorical.tolist()}, continuous {continuous.tolist()}")
    return categorical, continuous


class RAM_Mask:
    """Per env, the categorical and continuous RAM bytes RND reads.

    Every (game, byte) pair is its own field, so a slot shared by envs of different games never mixes the
    meaning of different bytes: categorical fields own their own RAM_VALUES embedding rows, continuous fields
    their own running statistics. Games with fewer bytes of a kind are padded to the longest mask of that kind,
    so envs of different games can share the RND batch.
    """

    def __init__(self, game_ids, ram_size, ram_masks=None):
        ram_masks = RND_RAM_MASKS if ram_masks is None else ram_masks
        # games without a curated mask read their full RAM, so their states are practically always new
        self.unmasked_games = [name for name in dict.fromkeys(game_name(gid) for gid in game_ids) if name not in ram_masks]
        game_fields = {}
        self.num_categorical_fields = 0
        self.num_continuous_fields = 0
        for name in dict.fromkeys(game_name(gid) for gid in game_ids):
            categorical, continuous = parse_ram_mask(ram_masks.get(name), ram_size, name)
            game_fields[name] = (
                categorical, np.arange(len(categorical)) + self.num_categorical_fields,
                continuous, np.arange(len(continuous)) + self.num_continuous_fields,
            )
            self.num_categorical_fields += len(categorical)
            self.num_continuous_fields += len(continuous)

        # token = field * RAM_VALUES + byte value; the last token is the padding
        self.num_tokens = self.num_categorical_fields * RAM_VALUES
        self.padding_token = self.num_tokens

        env_fields = [game_fields[game_name(gid)] for gid in game_ids]
        self.categorical_size = max(len(f[0]) for f in env_fields)
        self.continuous_size = max(len(f[2]) for f in env_fields)
        num_envs = len(env_fields)
        self.categorical_bytes = np.zeros((num_envs, self.categorical_size), dtype=np.int64)
        self.categorical_fields = np.full((num_envs, self.categorical_size), -1, dtype=np.int64)
        self.continuous_bytes = np.zeros((num_envs, self.continuous_size), dtype=np.int64)
        self.continuous_fields = np.full((num_envs, self.continuous_size), -1, dtype=np.int64)
        for i, (cat_bytes, cat_fields, cont_bytes, cont_fields) in enumerate(env_fields):
            self.categorical_bytes[i, :len(cat_bytes)] = cat_bytes
            self.categorical_fields[i, :len(cat_fields)] = cat_fields
            self.continuous_bytes[i, :len(cont_bytes)] = cont_bytes
            self.continuous_fields[i, :len(cont_fields)] = cont_fields


    def __call__(self, rams, env_indices):
        """rams: uint8 (N, ram_size) of the envs in env_indices ->
        - tokens: int64 (N, categorical_size), embedding rows of the categorical bytes, padding_token on padded slots
        - values: uint8 (N, continuous_size), the continuous bytes, 0 on padded slots
        - fields: int64 (N, continuous_size), the continuous field of each value, -1 on padded slots
        """
        env_indices = np.asarray(env_indices)
        rams = np.asarray(rams)

        cat_fields = self.categorical_fields[env_indices]
        cat_values = np.take_along_axis(rams, self.categorical_bytes[env_indices], axis=1).astype(np.int64)
        tokens = np.where(cat_fields >= 0, cat_fields * RAM_VALUES + cat_values, self.padding_token)

        fields = self.continuous_fields[env_indices]
        values = np.take_along_axis(rams, self.continuous_bytes[env_indices], axis=1) * (fields >= 0).astype(np.uint8)
        return tokens, values, fields


class Running_Mean_Std(nn.Module):
    """Batched (Chan et al.) running mean/variance kept as buffers so it is persisted with the model.
    The count is kept per element, so fields can be updated independently (update_fields).
    window: if set, the count is capped at it, so the statistics forget like an exponential moving average
    over about `window` samples instead of weighting the whole history equally."""

    def __init__(self, shape=(), window=None):
        super().__init__()
        self.window = window
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.full(shape, 1e-4, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x):
        x = x.to(torch.float64)
        self._combine(x.mean(dim=0), x.var(dim=0, unbiased=False), x.shape[0])

    @torch.no_grad()
    def update_fields(self, values, fields):
        """values, fields: 1-D. Updates field f (an index of the 1-D statistics) with the values whose field is f;
        fields without values are left unchanged."""
        values = values.to(torch.float64)
        zeros = torch.zeros_like(self.mean)
        batch_count = zeros.index_add(0, fields, torch.ones_like(values))
        batch_mean = zeros.index_add(0, fields, values) / batch_count.clamp(min=1)
        batch_var = zeros.index_add(0, fields, (values - batch_mean[fields]).pow(2)) / batch_count.clamp(min=1)
        self._combine(batch_mean, batch_var, batch_count)

    def _combine(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        m2 = self.var * self.count + batch_var * batch_count + delta.pow(2) * self.count * batch_count / total
        self.mean.copy_(self.mean + delta * batch_count / total)
        self.var.copy_(m2 / total)
        self.count.copy_(total if self.window is None else total.clamp(max=self.window))


class Episodic_Cell_Counter:
    """Per env, visit counts of RAM cells within the current episode: NGU's episodic novelty, with exact counts
    of curated RAM cells in place of a kNN over a learned embedding.

    A cell is the categorical bytes (room, level, items, ...) plus the continuous bytes (positions) binned by bin_size.
    The counts of an env are cleared at the end of its episode (reset), so the bonus never runs out across episodes.
    Also keeps the cells and the categorical states ever reached, as an exploration-progress diagnostic.
    """

    def __init__(self, num_envs, bin_size=8):
        self.bin_size = bin_size
        self.counts = [dict() for _ in range(num_envs)]
        self.cells_seen = set()
        self.categorical_seen = set()

    def visit(self, tokens, values, env_indices):
        """tokens, values: RAM_Mask outputs (numpy) of the envs in env_indices -> float64 (N,) 1 / sqrt(episodic visit count)"""
        cells = np.concatenate([tokens, values.astype(np.int64) // self.bin_size], axis=1)
        rewards = np.empty((len(env_indices),), dtype=np.float64)
        for k, i in enumerate(env_indices):
            key = cells[k].tobytes()
            n = self.counts[i].get(key, 0) + 1
            self.counts[i][key] = n
            rewards[k] = n ** -0.5
            self.cells_seen.add(key)
            self.categorical_seen.add(tokens[k].tobytes())
        return rewards

    def reset(self, env_index):
        self.counts[env_index].clear()


def mlp_trunk(in_size, hidden_size):
    # tf.nn.leaky_relu, used by the reference implementation, defaults to slope 0.2
    return nn.Sequential(
        nn.Linear(in_size, hidden_size), nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.2),
    )


class RND_Network(nn.Module):
    """Embeds each categorical token, concatenates the embeddings with the normalized continuous values and runs `body`.
    The padding token embeds to zeros and gets no gradient."""

    def __init__(self, num_tokens, embedding_size, body):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens + 1, embedding_size, padding_idx=num_tokens)
        self.body = body

    def forward(self, tokens, continuous):
        return self.body(torch.cat([self.embedding(tokens).flatten(1), continuous], dim=1))


class RND_Model(nn.Module, Safe_nn_Module):
    """Random Network Distillation (Burda et al., 2018), after openai/random-network-distillation, shared and batched across environments.

    - Input: the RAM of s', reduced to the bytes of its game's mask (RND_RAM_MASKS).
      Categorical bytes (room, level, items, ...) are embedded: every (game, byte, value) has its own embedding row,
      so an unseen value reaches the predictor through a row it has never trained, i.e. as maximally novel.
      Continuous bytes (positions, ...) are normalized per (game, byte) by running statistics and clipped to [-5, 5].
    - Target: frozen random embeddings + MLP -> linear(rep_size).
      Predictor: its own trained embeddings + same MLP -> 2 hidden ReLU layers -> linear(rep_size).
      Every linear layer uses orthogonal init with gain sqrt(2); embeddings use the default N(0, 1),
      the scale of the normalized continuous values.
    - r_intrinsic = mean_k (predictor(s')_k - target(s')_k)^2 / running_std(per-env discounted intrinsic return)
    - The predictor trains every step on a random `update_proportion` of the batch, so its learning speed
      does not grow with the number of envs.
    - episodic=True (NGU, Badia et al., 2020): RND only modulates an episodic novelty that never decays across episodes,
      r_intrinsic = r_episodic * clip(1 + (error - mean(error)) / std(error), 1, max_modulation),
      with r_episodic = 1 / sqrt(visits of the RAM cell of s' in this episode) (Episodic_Cell_Counter).
      Needs a curated RND_RAM_MASKS entry for every game. Call reset_episode(i) when env i's episode ends.
    - stats_window_steps: if set, the reward (and error) statistics forget like a moving average over about that many
      steps; otherwise the large errors of early training keep shrinking the normalized reward for a long time.
    """

    def __init__(
        self,
        game_ids,
        ram_size: int = 128,
        ram_masks: dict = None,
        embedding_size: int = 16,
        trunk_hidden_size: int = 256,
        rep_size: int = 256,
        predictor_hidden_size: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        update_proportion: float = None,
        warmup_steps: int = 128,
        episodic: bool = False,
        cell_bin_size: int = 8,
        max_modulation: float = 5.0,
        stats_window_steps: int = None,
        device="cpu",
        persistence_path=None,
        seed: int = 0,
    ):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="rnd_model", device=device, persistence_path=persistence_path)
        num_envs = len(game_ids)
        self.ram_mask = ram_mask = RAM_Mask(game_ids, ram_size, ram_masks)
        self.obs_size = obs_size = ram_mask.categorical_size * embedding_size + ram_mask.continuous_size
        self.gamma = gamma
        # the reference keeps the predictor's effective batch at that of 32 envs as the env count grows
        self.update_proportion = min(1.0, 32.0 / num_envs) if update_proportion is None else update_proportion
        # rewards stay 0 until the statistics have seen this many states: the first batches are near-identical
        # start states, whose tiny variance would blow up the normalized reward
        self.warmup_states = warmup_steps * num_envs
        if stats_window_steps is not None and stats_window_steps < warmup_steps:
            # the warmup is read off the (capped) count, so a shorter window would never end it
            raise ValueError(f"stats_window_steps ({stats_window_steps}) must be at least warmup_steps ({warmup_steps})")
        if episodic and ram_mask.unmasked_games:
            raise ValueError(f"Episodic novelty needs a curated RND_RAM_MASKS entry, missing for {ram_mask.unmasked_games}")
        self.episodic = episodic
        self.max_modulation = max_modulation
        # without a curated mask every full-RAM cell is new, so the counts would only grow
        self.cell_counter = None if ram_mask.unmasked_games else Episodic_Cell_Counter(num_envs, cell_bin_size)
        self.diagnostic_sums = {}
        self.diagnostic_count = 0
        # private RNGs so enabling RND does not shift the global streams used by the policy and the envs
        # (torch.manual_seed would also reseed CUDA, which fork_rng(devices=[]) does not restore)
        self.rng = np.random.default_rng(seed)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            # 1. Fixed random target network (weights frozen permanently)
            self.target_net = RND_Network(ram_mask.num_tokens, embedding_size, nn.Sequential(
                mlp_trunk(obs_size, trunk_hidden_size),
                nn.Linear(trunk_hidden_size, rep_size),
            ))
            # 2. Predictor network (trained online to match target network output)
            self.predictor_net = RND_Network(ram_mask.num_tokens, embedding_size, nn.Sequential(
                mlp_trunk(obs_size, trunk_hidden_size),
                nn.Linear(trunk_hidden_size, predictor_hidden_size), nn.ReLU(),
                nn.Linear(predictor_hidden_size, predictor_hidden_size), nn.ReLU(),
                nn.Linear(predictor_hidden_size, rep_size),
            ))
            # Walk the networks directly
            for m in list(self.target_net.modules()) + list(self.predictor_net.modules()):
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.zeros_(m.bias)
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.obs_stats = Running_Mean_Std((ram_mask.num_continuous_fields,))
        stats_window = None if stats_window_steps is None else stats_window_steps * num_envs
        self.reward_stats = Running_Mean_Std(window=stats_window)
        if episodic:
            # registered only in episodic mode, so pure-RND checkpoints keep loading
            self.error_stats = Running_Mean_Std(window=stats_window)
        # discounted intrinsic return per env, never reset on episode end (intrinsic return is non-episodic)
        self.register_buffer("reward_filter", torch.zeros((num_envs,), dtype=torch.float64), persistent=False)
        self.to(device)

        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=lr)
        # separate saver: Safe_nn_Module.load stops after the first module of a multi-module dict
        self.learner = Safe_nn_Module(
            name="rnd_learner", device=device, persistence_path=persistence_path,
            modules={"rnd_learner": self.optimizer}
        )

        self.load()
        # Adam's state_dict carries the lr it was saved with; the configured lr wins on resume
        for group in self.optimizer.param_groups:
            group["lr"] = lr


    def _normalize(self, values, fields):
        """values, fields: (N, continuous_size) -> float32 whitened values, 0 on padded slots."""
        valid = fields >= 0
        fields = fields.clamp(min=0)
        mean = self.obs_stats.mean[fields]
        std = self.obs_stats.var[fields].sqrt().clamp(min=1e-8)
        normalized = ((values.to(torch.float64) - mean) / std).clamp(-5.0, 5.0)
        return torch.where(valid, normalized, 0.0).to(torch.float32)


    def compute_intrinsic_rewards(self, rams, env_indices):
        """
        rams: uint8 array (N, ram_size), the RAM of s' for each env that stepped.
        env_indices: the env index of each state, for the per-env discounted intrinsic return.
        Returns np array (N,) of normalized intrinsic rewards and trains the predictor one step.
        """
        if len(env_indices) == 0:
            return np.zeros((0,), dtype=np.float32)

        tokens_np, values_np, fields_np = self.ram_mask(rams, env_indices)
        tokens = torch.as_tensor(tokens_np, device=self.device)
        values = torch.as_tensor(values_np, device=self.device)
        fields = torch.as_tensor(fields_np, device=self.device)
        # stats include the current batch, so the very first batch is normalized by its own statistics
        valid = fields >= 0
        self.obs_stats.update_fields(values[valid], fields[valid])
        x = self._normalize(values, fields)

        with torch.no_grad():
            target_out = self.target_net(tokens, x)
            raw_r_i = (self.predictor_net(tokens, x) - target_out).pow(2).mean(dim=1)

        keep = np.nonzero(self.rng.random(x.shape[0]) < self.update_proportion)[0]
        if len(keep) > 0:
            keep = torch.as_tensor(keep, dtype=torch.long, device=self.device)
            loss = (self.predictor_net(tokens[keep], x[keep]) - target_out[keep]).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        raw_r_i = raw_r_i.to(torch.float64)
        idx = torch.as_tensor(np.asarray(env_indices), dtype=torch.long, device=self.device)
        self.reward_filter[idx] = self.reward_filter[idx] * self.gamma + raw_r_i
        self.reward_stats.update(self.reward_filter[idx])
        warmed_up = self.reward_stats.count.item() >= self.warmup_states
        if warmed_up:
            rnd_reward = raw_r_i / torch.sqrt(self.reward_stats.var + 1e-8)
        else:
            rnd_reward = torch.zeros_like(raw_r_i)

        episodic_reward = None
        if self.cell_counter is not None:
            episodic_reward = torch.as_tensor(self.cell_counter.visit(tokens_np, values_np, env_indices), device=self.device)

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
            len(env_indices),
            rnd_error=raw_r_i, rnd_reward=rnd_reward, episodic_reward=episodic_reward,
            modulator=modulator, intrinsic_reward=reward,
        )
        return reward.cpu().numpy().astype(np.float32)


    def reset_episode(self, env_index):
        """Call when env_index's episode ends: clears its episodic counts. The RND state itself is non-episodic."""
        if self.cell_counter is not None:
            self.cell_counter.reset(env_index)


    def _accumulate_diagnostics(self, n, **tensors):
        # summed on the device; synced only in pop_diagnostics
        for key, t in tensors.items():
            if t is not None:
                self.diagnostic_sums[key] = self.diagnostic_sums.get(key, 0.0) + t.sum()
        self.diagnostic_count += n


    def pop_diagnostics(self):
        """Per-state means since the last call (rnd_error, rnd_reward, episodic_reward, modulator, intrinsic_reward)
        and the number of distinct cells / categorical states reached since start; resets the means."""
        n = max(self.diagnostic_count, 1)
        out = {key: float(v) / n for key, v in self.diagnostic_sums.items()}
        if self.cell_counter is not None:
            out["cells_seen"] = len(self.cell_counter.cells_seen)
            out["categorical_seen"] = len(self.cell_counter.categorical_seen)
        self.diagnostic_sums = {}
        self.diagnostic_count = 0
        return out


    def save(self, num_to_keep=2, override_persistence_path=None):
        Safe_nn_Module.save(self, num_to_keep=num_to_keep, override_persistence_path=override_persistence_path)
        self.learner.save(num_to_keep=num_to_keep, override_persistence_path=override_persistence_path)


    def load(self, override_persistence_path=None):
        Safe_nn_Module.load(self, override_persistence_path=override_persistence_path)
        self.learner.load(override_persistence_path=override_persistence_path)
        # Adam's load_state_dict does not check shapes, so a checkpoint saved with another obs_size would
        # only fail at the next step; the network itself then failed to load, so start the optimizer fresh too
        for p in self.predictor_net.parameters():
            if any(torch.is_tensor(v) and v.dim() > 0 and v.shape != p.shape for v in self.optimizer.state.get(p, {}).values()):
                logging.warning("RND optimizer checkpoint does not match the predictor shapes, discarding it.")
                self.optimizer.state.clear()
                break
