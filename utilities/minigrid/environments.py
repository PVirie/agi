import gymnasium as gym
from minigrid.wrappers import ActionBonus
import numpy as np
import os
import csv
import time


MINIGRID_ACTIONS = [
    "left", "right", "forward", "pickup", "drop", "toggle", "done"
]


class Multi_Environment:
    def __init__(self, 
        game_ids,
        tokenizer,
        record_statistic_dir=None
        ):

        self.tokenizer = tokenizer

        self.game_ids = game_ids
        self.record_statistic_dir = record_statistic_dir
        if record_statistic_dir is not None:
            os.makedirs(record_statistic_dir, exist_ok=True)
        self.steps = 0
        self.save_step_interval = 1000
        self.last_batch_episode_returns = [0 for _ in game_ids] # List to store the return of the last episode for each env
        self.last_batch_episode_lengths = [0 for _ in game_ids] # List to store the length of the last episode for each env
        self.last_batch_episode_times = [0 for _ in game_ids] # List to store the time taken for the last episode for each env
        self.batch_episode_returns = [[] for _ in game_ids] # List of lists to store episode returns for each env
        self.batch_episode_lengths = [[] for _ in game_ids] # List of lists to store episode lengths for each env
        self.batch_episode_times = [[] for _ in game_ids] # List of lists to store episode times for each env

        # 1. Generate Available Actions List
        # Structure: List[List[int]] -> [[0, 2, 3], [0, 1, 4, 5], ...]
        self.available_actions = []
        self.envs = []
        for gid in game_ids:
            self.available_actions.append(list(range(len(MINIGRID_ACTIONS)))) 
            env = gym.make(gid)

            self.envs.append(env)

        total = len(self.envs)
        self.return_obs = [None] * total
        self.return_rewards = [0] * total
        self.return_terminations = [False] * total
        self.return_truncations = [False] * total
        self.return_infos = [None] * total

        self.total_return = [0] * total
        self.total_length = [0] * total
        self.total_duration = [0] * total


    def __record_episode_statistics(self, batch):
        r = self.total_return[batch]
        l = self.total_length[batch]
        t = time.perf_counter() - self.total_duration[batch]
        self.total_return[batch] = 0
        self.total_length[batch] = 0
        self.total_duration[batch] = time.perf_counter()
        self.last_batch_episode_returns[batch] = r
        self.last_batch_episode_lengths[batch] = l
        self.last_batch_episode_times[batch] = t

    
    def __save_episode_statistics(self):
        if self.record_statistic_dir is None:
            return

        for i in range(len(self.game_ids)):
            self.batch_episode_returns[i].append(self.last_batch_episode_returns[i])
            self.batch_episode_lengths[i].append(self.last_batch_episode_lengths[i])
            self.batch_episode_times[i].append(self.last_batch_episode_times[i])

        if self.steps % self.save_step_interval == 0:
            # create a csv file for each env
            stat_file_name = os.path.join(self.record_statistic_dir, f"episode_statistics.csv")
            if not os.path.exists(stat_file_name):
                with open(stat_file_name, mode='w', newline='') as stat_file:
                    writer = csv.writer(stat_file)
                    header = []
                    for gid in self.game_ids:
                        header.extend([f"{gid}/return", f"{gid}/length", f"{gid}/time"])
                    writer.writerow(header)
            # append new stats
            with open(stat_file_name, mode='a', newline='') as stat_file:
                writer = csv.writer(stat_file)
                for i in range(len(self.batch_episode_returns[0])): # assuming all batches have same number of episodes
                    row = []
                    for j in range(len(self.game_ids)):
                        row.extend([
                            self.batch_episode_returns[j][i],
                            self.batch_episode_lengths[j][i],
                            self.batch_episode_times[j][i]
                        ])
                    writer.writerow(row)
            # clear batch stats after saving            
            self.batch_episode_returns = [[] for _ in self.game_ids]
            self.batch_episode_lengths = [[] for _ in self.game_ids]
            self.batch_episode_times = [[] for _ in self.game_ids]
                

    def reset(self, seed=None):
            # return self.envs.reset(seed=seed)
        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=seed)
            self.return_obs[i] = self.obs_to_object(obs)
            self.return_infos[i] = info

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
            self.return_infos[i] = info
            if termination or truncation:
                # record episode return and length
                self.__record_episode_statistics(i)
                obs, _ = env.reset() # You must manually reset!
                self.return_infos[i] = {
                    "episode": {
                        "r": self.last_batch_episode_returns[i],
                        "l": self.last_batch_episode_lengths[i],
                        "t": self.last_batch_episode_times[i]
                    }
                }
            self.return_obs[i] = self.obs_to_object(obs)
        self.steps += 1
        self.__save_episode_statistics()       
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

        self.steps = 0

        self.return_obs = None
        self.return_rewards = None
        self.return_terminations = None
        self.return_truncations = None
        self.return_infos = None

        self.total_return = None
        self.total_length = None
        self.total_duration = None


    def obs_to_object(self, obs, mission_max_len=32):
        # obs is a dict with keys: 'image', 'mission', 'direction'
        direction = obs['direction']  # scalar in [0, 3]
        image = obs['image']  # shape (7, 7, 3)
        mission = obs['mission']  # string

        output = np.zeros((1 + 7 * 7 * 3 + mission_max_len,), dtype=np.int32)
        output[0] = direction
        output[1:1 + 7 * 7 * 3] = image.flatten()
        mission_tokens = np.array(self.tokenizer([mission])[0], dtype=np.int32)
        # now padd to mission_max_len
        if len(mission_tokens) > mission_max_len:
            mission_tokens = mission_tokens[:mission_max_len]
        else:
            mission_tokens = np.pad(mission_tokens, (0, mission_max_len - len(mission_tokens)), constant_values=self.tokenizer.pad_token_id)
        output[1 + 7 * 7 * 3:] = mission_tokens
        return output
