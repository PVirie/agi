import abc
from typing import List, Any


class RL_Learner(abc.ABC):

    @abc.abstractmethod
    def reset(self, time = 0.0):
        pass

    @abc.abstractmethod
    def learn(self, 
              obs: Any, actions: Any, rewards: List[Any], 
              next_dones: List[List[bool]],
              masks: Any = None, valid_actions: Any = None):
        pass


class Supervised_Learner(abc.ABC):

    @abc.abstractmethod
    def train(self, obs: Any, actions: Any, target_actions: Any, valid_actions: Any = None, trained_logprob_indices: List[int] = None):
        pass