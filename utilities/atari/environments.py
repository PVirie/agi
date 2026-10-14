import ale_py
import gymnasium as gym
import numpy as np
import time

from .rnd import RND_Model

gym.register_envs(ale_py)

# Full Atari action set, in ALE's action order (Order matters!)
ATARI_ACTIONS = [
    "NOOP", "FIRE", "UP", "RIGHT", 
    "LEFT", "DOWN", "UPRIGHT", "UPLEFT", 
    "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", 
    "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", 
    "DOWNRIGHTFIRE", "DOWNLEFTFIRE"
]


class Ram_Preprocessing(gym.Wrapper):
    """The RAM counterpart of AtariPreprocessing, which cannot be used here because it replaces the
    observation with the screen. Frame skipping is left to ALE (v5 default: 4 frames, sticky actions).

    - reset: a random number (1..noop_max) of NOOP steps for start-state diversity.
    - terminal_on_life_loss: report termination when a life is lost; the next reset restarts the game.
    """

    def __init__(self, env, noop_max=30, terminal_on_life_loss=True):
        super().__init__(env)
        self.noop_max = noop_max
        self.terminal_on_life_loss = terminal_on_life_loss
        self.ale = env.unwrapped.ale
        self.lives = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.noop_max > 0:
            noops = self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
            for _ in range(noops):
                obs, _, termination, truncation, info = self.env.step(0)
                if termination or truncation:
                    obs, info = self.env.reset()
        self.lives = self.ale.lives()
        return obs, info

    def step(self, action):
        obs, reward, termination, truncation, info = self.env.step(action)
        if self.terminal_on_life_loss:
            lives = self.ale.lives()
            termination = termination or lives < self.lives
            self.lives = lives
        return obs, reward, termination, truncation, info


class Multi_Atari_Environment:
    def __init__(self, 
        game_ids,
        episodic_life=False,
        reward_clipping=True,
        prepend_objective=False,
        rnd_reward=False,
        rnd_weight=1.0,
        rnd_lr=1e-4,
        device="cpu",
        persistence_path=None,
        render_mode=None,
        ):

        self.game_ids = game_ids
        self.prepend_objective = prepend_objective
        self.reward_clipping = reward_clipping

        # the objective is a single uint8 byte, like each RAM byte: the index of the game id in order of first appearance
        unique_game_ids = list(dict.fromkeys(game_ids))
        if len(unique_game_ids) > 256:
            raise ValueError(f"At most 256 unique games fit in a uint8 objective, got {len(unique_game_ids)}")
        self.num_objectives = len(unique_game_ids)
        self.objective_size = 1
        self.game_id_to_objective = {
            gid: np.array([k], dtype=np.uint8) for k, gid in enumerate(unique_game_ids)
        }

        # Structure: List[List[int]] -> [[0, 2, 3], [0, 1, 4, 5], ...]
        self.available_actions = []
        self.envs = []
        for gid in game_ids:
            # ale-py >= 0.9 has no "-ram" ids: RAM is selected with obs_type, giving the 128-byte console RAM (uint8)
            env = gym.make(gid, obs_type="ram", full_action_space=True, render_mode=render_mode)
            # ALE action values are indices into the full action set, i.e. into ATARI_ACTIONS
            self.available_actions.append([int(a) for a in env.unwrapped.ale.getMinimalActionSet()])
            env = Ram_Preprocessing(env, noop_max=30, terminal_on_life_loss=episodic_life)
            self.envs.append(env)

        self.ram_size = self.envs[0].observation_space.shape[0]
        self.observation_size = self.ram_size + (self.objective_size if prepend_objective else 0)

        # one RND model shared by all envs: novelty is global. It reads the RAM of s' (masked per game), without the objective.
        self.rnd_weight = rnd_weight
        self.rnd = None
        if rnd_reward:
            self.rnd = RND_Model(
                game_ids=game_ids,
                ram_size=self.ram_size,
                lr=rnd_lr, device=device, persistence_path=persistence_path
            )

        total = len(self.envs)
        self.return_obs = [None] * total
        self.return_rewards = [0] * total
        self.return_terminations = [False] * total
        self.return_truncations = [False] * total
        self.return_infos = [None] * total

        self.total_return = [0] * total
        self.total_aux_return = [0] * total
        self.total_length = [0] * total
        self.cumulative_length = [0] * total # agent-env transitions since the run started; never reset per episode
        self.total_duration = [0] * total # start time of the episode for each env, misnoming but consistent with other statistics


    def prepare_observation(self, obs, env_index):
        if self.prepend_objective:
            objective = self.game_id_to_objective[self.game_ids[env_index]]
            return np.concatenate([objective, obs])
        else:
            return obs


    def reset(self, seed=None):
        for i, env in enumerate(self.envs):
            # offset per env: a shared seed would make every env replay the same episode
            obs, info = env.reset(seed=None if seed is None else seed + i)
            self.return_obs[i] = self.prepare_observation(obs, i)
            self.return_infos[i] = info

            self.total_return[i] = 0
            self.total_aux_return[i] = 0
            self.total_length[i] = 0
            self.cumulative_length[i] = 0
            self.total_duration[i] = time.perf_counter()
        return self.return_obs, self.return_infos


    def step(self, actions):
        stepped = []
        for i, env in enumerate(self.envs):
            if actions[i] is None:
                self.return_rewards[i] = 0
                self.return_terminations[i] = False
                self.return_truncations[i] = False
                self.return_infos[i] = {}
                continue
            obs, reward, termination, truncation, info = env.step(actions[i])
            stepped.append((i, obs, reward, termination, truncation, info))

        # novelty of s' is measured on its RAM, before any auto-reset replaces it
        intrinsic_rewards = np.zeros((len(stepped),), dtype=np.float32)
        if self.rnd is not None and len(stepped) > 0:
            intrinsic_rewards = self.rnd.compute_intrinsic_rewards(
                np.stack([obs for _, obs, *_ in stepped]),
                [i for i, *_ in stepped]
            )

        for (i, obs, reward, termination, truncation, info), r_intrinsic in zip(stepped, intrinsic_rewards):
            env = self.envs[i]
            # clipping applies to the training reward only; the episode statistics keep the true game score
            game_reward = float(reward)
            extrinsic_reward = float(np.clip(game_reward, -1.0, 1.0)) if self.reward_clipping else game_reward
            total_reward = extrinsic_reward + self.rnd_weight * float(r_intrinsic)

            info["original_reward"] = extrinsic_reward
            info["game_reward"] = game_reward
            info["intrinsic_reward"] = float(r_intrinsic)

            self.return_rewards[i] = total_reward
            self.return_terminations[i] = termination
            self.return_truncations[i] = truncation
            self.return_infos[i] = info

            self.total_return[i] += game_reward
            self.total_aux_return[i] += total_reward
            self.total_length[i] += 1
            self.cumulative_length[i] += 1
            if termination or truncation:
                info["episode"] = {
                    "r": self.total_return[i],
                    "aug_r": self.total_aux_return[i],
                    "l": self.total_length[i],
                    "e": self.cumulative_length[i],
                    "t": time.perf_counter() - self.total_duration[i],
                    # same convention as MiniGrid: an episode with positive game score counts as a success
                    "s": 1.0 if self.total_return[i] > 0 else 0.0
                }
                obs, _ = env.reset() # You must manually reset!

                self.total_return[i] = 0
                self.total_aux_return[i] = 0
                self.total_length[i] = 0
                self.total_duration[i] = time.perf_counter()

            self.return_obs[i] = self.prepare_observation(obs, i)
        return self.return_obs, self.return_rewards, self.return_terminations, self.return_truncations, self.return_infos


    def save(self):
        if self.rnd is not None:
            self.rnd.save()


    def total_env_steps(self):
        return sum(self.cumulative_length)


    def get_available_actions(self):
        """
        Returns a list of lists containing valid action indices for each env.
        Example: [[0, 2, 3], [0, 1, 4, 5]]
        """
        return self.available_actions


    def sample_valid_actions(self):
        """
        Helper: Randomly samples ONLY valid actions for each game.
        """
        actions = []
        for valid_indices in self.available_actions:
            # Pick one random index from the list of valid ones
            actions.append(np.random.choice(valid_indices))
        return np.array(actions)


    def close(self):
        for env in self.envs:
            env.close()

        self.return_obs = None
        self.return_rewards = None
        self.return_terminations = None
        self.return_truncations = None
        self.return_infos = None

        self.total_return = None
        self.total_aux_return = None
        self.total_length = None
        self.cumulative_length = None
        self.total_duration = None