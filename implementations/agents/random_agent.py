import random
import logging

from interfaces.agent import Agent


class Random_Agent(Agent):

    def __init__(self, id: str):
        self.id = id


    def reset(self):
        logging.info(f"Random agent {self.id} reset.")


    def choose_action(self, 
                      last_idles, next_dones, last_truncates, last_resets, 
                      latest_frames, rewards, next_available_actions, 
                      force_train=False):
        logging.info(f"Random agent {self.id} choosing action...")
        actions = []
        for d in next_dones:
            if random.random() < 0.5:
                actions.append([random.randint(0, 6) if not d else -1, random.randint(0, 63), random.randint(0, 63)])
            else:
                actions.append(None)
        return actions