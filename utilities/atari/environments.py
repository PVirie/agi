import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    AtariPreprocessing,
    TransformReward,
    FrameStackObservation,
    ResizeObservation,
    RecordEpisodeStatistics,
    NormalizeReward
)

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
        reward_clipping=True, # In this context, this flag enables Normalization
        render_mode=None):

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

        def make_env(game_id):
            def _init():
                env = gym.make(game_id, full_action_space=True, render_mode=render_mode)

                # 1. Tracker: Records the "True" game score before any clipping/normalization.
                # NOTE: Because episodic_life=True is used below, this will record the 
                # score per "Life", not per full "Game" (e.g., 3 lives). 
                # To log full game scores, you usually need a separate Eval env with episodic_life=False.
                env = RecordEpisodeStatistics(env)

                # 2. Preprocessing
                env = AtariPreprocessing(
                    env,
                    frame_skip=1,
                    grayscale_obs=grayscale,
                    scale_obs=False, # returns uint8 (0-255). Ensure your Agent divides by 255.0!
                    terminal_on_life_loss=episodic_life,
                    noop_max=30
                )

                # 3. Resize
                if img_height != 84 or img_width != 84:
                    env = ResizeObservation(env, (img_height, img_width))

                # 4. PPO Reward Processing
                if reward_clipping:
                    # A. Normalize: Tracks running mean/std of rewards
                    env = NormalizeReward(env, gamma=0.99)
                    
                    # B. Clip Normalized Reward (CRITICAL FIX)
                    # NormalizeReward can still output huge values (e.g. 50 sigma).
                    # PPO requires clipping the *normalized* reward, usually to [-10, 10].
                    env = TransformReward(env, lambda r: np.clip(r, -10, 10))

                # 5. Frame Stacking
                if stack_num > 1:
                    env = FrameStackObservation(env, stack_num)
                
                return env
            return _init

        env_fns = [make_env(gid) for gid in game_ids]
        
        self.envs = gym.vector.AsyncVectorEnv(env_fns)


    def sample_actions(self):
        """
        Returns a batch of random actions, one for each environment.
        Output shape: (num_envs, )
        """
        actions = []
        for valid_indices in self.available_actions:
            action = np.random.choice(valid_indices)
            actions.append(action)
        return np.array(actions)

    def reset(self, seed=None):
        return self.envs.reset(seed=seed)

    def step(self, actions, mask=None):
        return self.envs.step(actions)
    
    def get_real_final_observation(self, next_obs, infos, env_index):
        if "_final_observation" in infos and infos["_final_observation"][env_index]:
            return infos["final_observation"][env_index]
        return next_obs[env_index]

    def close(self):
        self.envs.close()