import abc
from typing import List, Any

from .core import Context_Collector

class Learner(abc.ABC):

    @abc.abstractmethod
    def reset(self, time = 0.0):
        pass

    @abc.abstractmethod
    def learn(self, obs: Context_Collector, actions: List[Any], logprobs: List[Any], rewards: List[List[float]], values: List[Any], next_dones: List[List[bool]], last_value: Any, last_done: List[bool]):
        pass