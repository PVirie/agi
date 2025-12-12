import random
import logging


class Random_Agent:

    def __init__(self, id: str):
        self.id = id


    def choose_action(self, latest_frames, dones, scores, next_available_actions):
        logging.info(f"Random agent {self.id} choosing action...")
        actions = []
        for d in dones:
            actions.append([random.randint(0, 6) if not d else -1, random.randint(0, 63), random.randint(0, 63)])
        return actions