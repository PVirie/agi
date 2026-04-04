import gymnasium as gym

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