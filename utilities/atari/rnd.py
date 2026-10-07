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


class RND_Model(nn.Module, Safe_nn_Module):
    """Random Network Distillation (Burda et al., 2018) intrinsic reward, shared and batched across environments.

    r_intrinsic = || predictor(s') - target(s') ||^2 / running_std(r_intrinsic)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_size: int = 128,
        feature_size: int = 64,
        lr: float = 1e-4,
        device="cpu",
        persistence_path=None,
        seed: int = 0,
    ):
        nn.Module.__init__(self)
        Safe_nn_Module.__init__(self, name="rnd_model", device=device, persistence_path=persistence_path)
        self.obs_dim = obs_dim

        # private RNG so enabling RND does not shift the global stream used to initialize the policy
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            # 1. Fixed random target network (weights frozen permanently)
            self.target_net = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, feature_size),
            )
            # 2. Predictor network (trained online to match target network output)
            self.predictor_net = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, feature_size),
            )
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.reward_stats = Running_Mean_Std()
        self.to(device)

        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=lr)
        # separate saver: Safe_nn_Module.load stops after the first module of a multi-module dict
        self.learner = Safe_nn_Module(
            name="rnd_learner", device=device, persistence_path=persistence_path,
            modules={"rnd_learner": self.optimizer}
        )

        self.load()


    def compute_intrinsic_rewards(self, observations):
        """
        observations: list of np arrays (any shape, flattened internally), one per environment.
        Returns np array of normalized intrinsic rewards (one per observation) and trains the predictor one step.
        """
        if len(observations) == 0:
            return np.zeros((0,), dtype=np.float32)

        flat_obs = torch.as_tensor(
            np.stack([np.asarray(o, dtype=np.float32).reshape(-1) for o in observations]),
            dtype=torch.float32, device=self.device
        )

        with torch.no_grad():
            target_out = self.target_net(flat_obs)
        pred_out = self.predictor_net(flat_obs)
        errors = (pred_out - target_out).pow(2).mean(dim=1)

        self.optimizer.zero_grad()
        errors.mean().backward()
        self.optimizer.step()

        raw_r_i = errors.detach()
        self.reward_stats.update(raw_r_i)
        normalized = raw_r_i.to(torch.float64) / torch.sqrt(self.reward_stats.var + 1e-8)
        return normalized.cpu().numpy().astype(np.float32)


    def save(self, num_to_keep=2, override_persistence_path=None):
        Safe_nn_Module.save(self, num_to_keep=num_to_keep, override_persistence_path=override_persistence_path)
        self.learner.save(num_to_keep=num_to_keep, override_persistence_path=override_persistence_path)


    def load(self, override_persistence_path=None):
        Safe_nn_Module.load(self, override_persistence_path=override_persistence_path)
        self.learner.load(override_persistence_path=override_persistence_path)
