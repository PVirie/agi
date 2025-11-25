import abc

class Learner(abc.ABC):

    @abc.abstractmethod
    def reset(self, time = 0.0):
        pass

    @abc.abstractmethod
    def collect(self, obs, value, action, logprob, reward, termination, truncation):
        pass

    @abc.abstractmethod
    def learn(self, last_value, last_termination, last_truncation):
        pass