import abc
from typing import List, Any

class Learner(abc.ABC):

    @abc.abstractmethod
    def reset(self, time = 0.0):
        pass

    @abc.abstractmethod
    def learn(self, obs, actions, logprobs, rewards, values, next_dones: List[bool], last_value: float, last_done: bool):
        pass