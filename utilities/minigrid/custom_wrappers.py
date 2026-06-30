import gymnasium as gym


class OriginalRewardWrapper(gym.Wrapper):
    """Stores the pre-bonus reward in info so it survives through PositionBonus."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info is None:
            info = {}
        info['original_reward'] = reward
        return obs, reward, terminated, truncated, info


# create a wrapper that takes the inventory information and adds it to the observation
class InventoryWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        inventory = self.env.unwrapped.carrying
        # inventory has type 
        if inventory:
            obs["inventory"] = f"{inventory.color} {inventory.type}"
        else:
            obs["inventory"] = ""
        return obs