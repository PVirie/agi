import abc

class Agent_Core(abc.ABC):

    @abc.abstractmethod
    def get_value(self, x):
        pass

    @abc.abstractmethod
    def get_action_and_value(self, x, action=None):
        pass