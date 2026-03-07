import abc
from typing import List, Any


class RL_Learner(abc.ABC):

    @abc.abstractmethod
    def update_learning_rate(self, time = 0.0):
        pass

    @abc.abstractmethod
    def learn(self, 
              obs: Any, last_actions: Any, rewards: List[Any], 
              next_dones: List[List[bool]],
              valid_actions: Any = None, masks: Any = None, aux_masks: Any = None):
        pass


class RL_Index_Learner(RL_Learner):

    @abc.abstractmethod
    def learn(self, 
              obs: Any, indices: Any,
              last_actions: Any, rewards: List[Any], 
              next_dones: List[List[bool]],
              valid_actions: Any = None, masks: Any = None, aux_masks: Any = None):
        pass
