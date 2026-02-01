import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    RecordEpisodeStatistics,
    AtariPreprocessing,
    FrameStackObservation,
    ResizeObservation,
    NormalizeReward,
    ClipReward
)

# Standard Atari Action Set (Order matters!)
ATARI_ACTIONS = [
    "NOOP", "FIRE", "UP", "RIGHT", 
    "LEFT", "DOWN", "UPRIGHT", "UPLEFT", 
    "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", 
    "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", 
    "DOWNRIGHTFIRE", "DOWNLEFTFIRE"
]

class Multi_Atari_Environment:
    def __init__(self, 
        game_ids,
        img_height=64,
        img_width=32,
        maxpool=True,
        stack_num=4,
        frameskip=4,
        grayscale=True,
        episodic_life=True,
        reward_clipping=True,
        render_mode=None):

        # 1. Generate Available Actions List
        # Structure: List[List[int]] -> [[0, 2, 3], [0, 1, 4, 5], ...]
        self.available_actions = []
        
        for gid in game_ids:
            # Initialize temp env with minimal action space to see what's valid
            temp_env = gym.make(gid, full_action_space=False)
            valid_action_names = temp_env.unwrapped.get_action_meanings()
            temp_env.close()

            # Convert names (e.g., "UP") to indices (e.g., 2)
            game_indices = []
            for name in valid_action_names:
                if name in ATARI_ACTIONS:
                    game_indices.append(ATARI_ACTIONS.index(name))
            
            self.available_actions.append(game_indices)

        # 2. Build Vector Env
        def make_env(game_id):
            def _init():
                env = gym.make(game_id, full_action_space=True, render_mode=render_mode)
                env = AtariPreprocessing(
                    env,
                    frame_skip=1,
                    grayscale_obs=grayscale,
                    scale_obs=False, # returns uint8 (0-255). Ensure your Agent divides by 255.0!
                    terminal_on_life_loss=episodic_life,
                    noop_max=30
                )
                
                env = RecordEpisodeStatistics(env)

                if img_height != 84 or img_width != 84:
                    env = ResizeObservation(env, (img_height, img_width))

                if reward_clipping:
                    # env = NormalizeReward(env, gamma=0.99)
                    env = ClipReward(env, min_reward=-1, max_reward=1)

                if stack_num > 1:
                    env = FrameStackObservation(env, stack_size=stack_num, padding_type='zero')
                
                return env
            return _init

        env_fns = [make_env(gid) for gid in game_ids]
        self.envs = gym.vector.AsyncVectorEnv(env_fns)


    def reset(self, seed=None):
        return self.envs.reset(seed=seed)


    def step(self, actions, mask=None):
        return self.envs.step(actions)


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


    def get_real_final_observation(self, next_obs, infos, env_index):
        if "_final_observation" in infos and infos["_final_observation"][env_index]:
            return infos["final_observation"][env_index]
        return next_obs[env_index]


    def close(self):
        self.envs.close()