import random
import logging


class Random_Agent:

    def __init__(self, id: str):
        self.id = id


    def choose_action(self, last_idles, next_dones, last_truncates, last_resets, latest_frames, scores, next_available_actions, force_train=False):
        logging.info(f"Random agent {self.id} choosing action...")
        actions = []
        for d in next_dones:
            if random.random() < 0.5:
                actions.append([random.randint(0, 6) if not d else -1, random.randint(0, 63), random.randint(0, 63)])
            else:
                actions.append(None)
        return actions