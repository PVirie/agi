import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
import numpy as np
import time

from .custom_wrappers import InventoryWrapper

MINIGRID_ACTIONS = [
    "left", "right", "forward", "pickup", "drop", "toggle", "done"
]


class Multi_Environment:
    def __init__(self, 
        game_ids,
        tokenizer,
        mission_max_len=32,
        full_mdp=False,
        full_mdp_width=64,
        full_mdp_height=64,
        include_inventory=True
        ):

        self.mission_max_len = mission_max_len
        self.full_mdp = full_mdp
        self.full_mdp_width = full_mdp_width
        self.full_mdp_height = full_mdp_height
        self.include_inventory = include_inventory

        self.tokenizer = tokenizer

        self.game_ids = game_ids

        # 1. Generate Available Actions List
        # Structure: List[List[int]] -> [[0, 2, 3], [0, 1, 4, 5], ...]
        self.available_actions = []
        self.envs = []
        for gid in game_ids:
            self.available_actions.append(list(range(len(MINIGRID_ACTIONS)))) 
            env = gym.make(gid)
            if include_inventory:
                env = InventoryWrapper(env)
            if full_mdp:
                env = FullyObsWrapper(env)
            self.envs.append(env)

        total = len(self.envs)
        self.return_obs = [None] * total
        self.return_rewards = [0] * total
        self.return_terminations = [False] * total
        self.return_truncations = [False] * total
        self.return_infos = [None] * total

        self.total_return = [0] * total
        self.total_length = [0] * total
        self.total_duration = [0] * total # store the start time of the episode for each env, misnoming but consistent with other statistics


    def reset(self, seed=None):
            # return self.envs.reset(seed=seed)
        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=seed)
            self.return_obs[i] = self.obs_to_object(obs)
            self.return_infos[i] = None

            self.total_return[i] = 0
            self.total_length[i] = 0
            self.total_duration[i] = time.perf_counter()
        return self.return_obs, self.return_infos


    def step(self, actions):
        for i, env in enumerate(self.envs):
            if actions[i] is None:
                self.return_rewards[i] = 0
                self.return_terminations[i] = False
                self.return_truncations[i] = False
                continue
            obs, reward, termination, truncation, info = env.step(actions[i])
            self.return_rewards[i] = reward
            self.return_terminations[i] = termination
            self.return_truncations[i] = truncation
            self.return_infos[i] = None

            self.total_return[i] += reward
            self.total_length[i] += 1
            if termination or truncation:
                obs, _ = env.reset() # You must manually reset!
                self.return_infos[i] = {
                    "episode": {
                        "r": self.total_return[i],
                        "l": self.total_length[i],
                        "t": time.perf_counter() - self.total_duration[i]
                    }
                }
                self.total_return[i] = 0
                self.total_length[i] = 0
                self.total_duration[i] = time.perf_counter()
            self.return_obs[i] = self.obs_to_object(obs)
        return self.return_obs, self.return_rewards, self.return_terminations, self.return_truncations, self.return_infos


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
        self.total_length = None
        self.total_duration = None


    def obs_to_object(self, obs):
        # obs is a dict with keys: 'image', 'mission', 'direction', 'inventory'
        # output (mission, direction, inventory, image) as a flat array
        # where image is padded to the top left corner of a full_mdp_height x full_mdp_width x 3 array, 
        # and the rest is 0, then flattened and concatenated with mission tokens

        mission = obs['mission']  # string
        mission_tokens = np.array(self.tokenizer([mission])[0], dtype=np.int32)
        # now padd to mission_max_len
        if len(mission_tokens) > self.mission_max_len:
            mission_tokens = mission_tokens[:self.mission_max_len]
        else:
            mission_tokens = np.pad(mission_tokens, (0, self.mission_max_len - len(mission_tokens)), constant_values=self.tokenizer.pad_token_id)
            
        direction = ['right', 'down', 'left', 'up'][obs['direction']]
        if self.include_inventory:
            inventory = obs['inventory']
            internal_state_tokens = np.array(self.tokenizer([direction, inventory])[0], dtype=np.int32)
        else:
            internal_state_tokens = np.array(self.tokenizer([direction])[0], dtype=np.int32)

        # pad internal_state_tokens to 3
        if len(internal_state_tokens) > 3:
            internal_state_tokens = internal_state_tokens[:3]
        else:
            internal_state_tokens = np.pad(internal_state_tokens, (0, 3 - len(internal_state_tokens)), constant_values=self.tokenizer.pad_token_id)
        
        if self.full_mdp:
            image = obs['image']  # shape (h, w, 3)
            # pad image to top left corner of a full_mdp_width x full_mdp_height x 3 array, the rest is 255,
            # then flatten it and concatenate with mission tokens
            image_padded = np.full((self.full_mdp_height, self.full_mdp_width, 3), 255, dtype=np.uint8)
            h, w, _ = image.shape
            image_padded[:h, :w, :] = image

            output = np.zeros((self.mission_max_len + 3 + (self.full_mdp_height * self.full_mdp_width * 3),), dtype=np.int32)
            output[:self.mission_max_len] = mission_tokens
            output[self.mission_max_len:self.mission_max_len + 3] = internal_state_tokens
            output[(self.mission_max_len + 3):] = image_padded.flatten()
        else:
            image = obs['image']  # shape (7, 7, 3)

            output = np.zeros((self.mission_max_len + 3 + (7 * 7 * 3),), dtype=np.int32)
            output[:self.mission_max_len] = mission_tokens
            output[self.mission_max_len:self.mission_max_len + 3] = internal_state_tokens
            output[(self.mission_max_len + 3):] = image_padded.flatten()
        return output
