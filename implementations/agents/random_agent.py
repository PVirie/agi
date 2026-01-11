import random
import logging
import numpy as np

from interfaces.agent import Agent


class Random_Agent(Agent):

    def __init__(self, id: str):
        self.id = id


    def reset(self):
        logging.info(f"Random agent {self.id} reset.")


    def choose_action(self, 
                      last_idles, last_dones, last_truncates, last_resets, 
                      latest_frames, rewards, next_available_actions, 
                      force_train=False):
        logging.info(f"Random agent {self.id} choosing action...")
        actions = []
        for _ in latest_frames:
            if random.random() < 0.5:
                actions.append(np.array([random.randint(0, 6), random.randint(0, 63), random.randint(0, 63)], dtype=np.int32))
            else:
                actions.append(None)
        return actions