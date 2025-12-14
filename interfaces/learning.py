import abc
from typing import List, Any


class PPO_Learner(abc.ABC):

    @abc.abstractmethod
    def reset(self, time = 0.0):
        pass

    @abc.abstractmethod
    def learn(self, obs: Any, actions: Any, logprobs: List[Any], values: List[Any], rewards: List[List[float]], next_dones: List[List[bool]], last_value: Any, last_done: List[bool], masks: Any = None):
        pass


class Supervised_Learner(abc.ABC):

    @abc.abstractmethod
    def train(self, obs: Any, actions: Any, target_actions: Any, masks: Any = None):
        pass