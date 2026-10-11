import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utilities.safe_torch_module import Safe_nn_Module


class Running_Mean_Std(nn.Module):
    """Batched (Chan et al.) running mean/variance kept as buffers so it is persisted with the model."""

    def __init__(self, shape=()):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x):
        x = x.to(torch.float64)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        m2 = self.var * self.count + batch_var * batch_count + delta.pow(2) * self.count * batch_count / total
        self.mean.copy_(self.mean + delta * batch_count / total)
        self.var.copy_(m2 / total)
        self.count.copy_(total)


def mlp_trunk(in_size, hidden_size):
    # tf.nn.leaky_relu, used by the reference implementation, defaults to slope 0.2
    return nn.Sequential(
        nn.Linear(in_size, hidden_size), nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.2),
    )


class RND_Model(nn.Module, Safe_nn_Module):
    """Random Network Distillation (Burda et al., 2018), after openai/random-network-distillation, shared and batched across environments.

    - Input: the RAM vector of s' (uint8, 128 bytes for Atari), normalized per byte by running statistics
      and clipped to [-5, 5].
    - Target: frozen random MLP -> linear(rep_size). Predictor: same MLP -> 2 hidden ReLU layers -> linear(rep_size).
      Every layer uses orthogonal init with gain sqrt(2).
    - r_intrinsic = mean_k (predictor(s')_k - target(s')_k)^2 / running_std(per-env discounted intrinsic return)
    - The predictor trains every step on a random `update_proportion` of the batch, so its learning speed
      does not grow with the number of envs.
    """

    def __init__(
        self,
        num_envs: int,
        obs_size: int = 128,
        trunk_hidden_size: int = 256,
        rep_size: int = 256,
        predictor_hidden_size: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        update_proportion: float = None,
        warmup_steps: int = 128,
        device="cpu",
        persistence_path=None,
        seed: int = 0,
    ):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="rnd_model", device=device, persistence_path=persistence_path)
        self.obs_size = obs_size
        self.gamma = gamma
        # the reference keeps the predictor's effective batch at that of 32 envs as the env count grows
        self.update_proportion = min(1.0, 32.0 / num_envs) if update_proportion is None else update_proportion
        # rewards stay 0 until the statistics have seen this many states: the first batches are near-identical
        # start states, whose tiny variance would blow up the normalized reward
        self.warmup_states = warmup_steps * num_envs

        # private RNGs so enabling RND does not shift the global streams used by the policy and the envs
        # (torch.manual_seed would also reseed CUDA, which fork_rng(devices=[]) does not restore)
        self.rng = np.random.default_rng(seed)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            # 1. Fixed random target network (weights frozen permanently)
            self.target_net = nn.Sequential(
                mlp_trunk(obs_size, trunk_hidden_size),
                nn.Linear(trunk_hidden_size, rep_size),
            )
            # 2. Predictor network (trained online to match target network output)
            self.predictor_net = nn.Sequential(
                mlp_trunk(obs_size, trunk_hidden_size),
                nn.Linear(trunk_hidden_size, predictor_hidden_size), nn.ReLU(),
                nn.Linear(predictor_hidden_size, predictor_hidden_size), nn.ReLU(),
                nn.Linear(predictor_hidden_size, rep_size),
            )
            # self.modules is shadowed by Safe_nn_Module's checkpoint dict, so walk the networks directly
            for m in list(self.target_net.modules()) + list(self.predictor_net.modules()):
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.zeros_(m.bias)
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.obs_stats = Running_Mean_Std((obs_size,))
        self.reward_stats = Running_Mean_Std()
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


    def _normalize(self, states):
        mean = self.obs_stats.mean.to(torch.float32)
        std = self.obs_stats.var.sqrt().clamp(min=1e-8).to(torch.float32)
        return ((states - mean) / std).clamp(-5.0, 5.0)


    def compute_intrinsic_rewards(self, states, env_indices):
        """
        states: uint8 array (N, obs_size), the RAM of s' for each env that stepped.
        env_indices: the env index of each state, for the per-env discounted intrinsic return.
        Returns np array (N,) of normalized intrinsic rewards and trains the predictor one step.
        """
        if len(env_indices) == 0:
            return np.zeros((0,), dtype=np.float32)

        x = torch.as_tensor(np.asarray(states), device=self.device).to(torch.float32).reshape(-1, self.obs_size)
        # stats include the current batch, so the very first batch is normalized by its own statistics
        self.obs_stats.update(x)
        x = self._normalize(x)

        with torch.no_grad():
            target_out = self.target_net(x)
            raw_r_i = (self.predictor_net(x) - target_out).pow(2).mean(dim=1)

        keep = np.nonzero(self.rng.random(x.shape[0]) < self.update_proportion)[0]
        if len(keep) > 0:
            keep = torch.as_tensor(keep, dtype=torch.long, device=self.device)
            loss = (self.predictor_net(x[keep]) - target_out[keep]).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        raw_r_i = raw_r_i.to(torch.float64)
        idx = torch.as_tensor(np.asarray(env_indices), dtype=torch.long, device=self.device)
        self.reward_filter[idx] = self.reward_filter[idx] * self.gamma + raw_r_i
        self.reward_stats.update(self.reward_filter[idx])
        if self.reward_stats.count.item() < self.warmup_states:
            return np.zeros((len(env_indices),), dtype=np.float32)

        normalized = raw_r_i / torch.sqrt(self.reward_stats.var + 1e-8)
        return normalized.cpu().numpy().astype(np.float32)


    def save(self, num_to_keep=2, override_persistence_path=None):
        Safe_nn_Module.save(self, num_to_keep=num_to_keep, override_persistence_path=override_persistence_path)
        self.learner.save(num_to_keep=num_to_keep, override_persistence_path=override_persistence_path)


    def load(self, override_persistence_path=None):
        Safe_nn_Module.load(self, override_persistence_path=override_persistence_path)
        self.learner.load(override_persistence_path=override_persistence_path)
