import abc

class Learner(abc.ABC):

    @abc.abstractmethod
    def reset(self, time = 0.0):
        pass

    @abc.abstractmethod
    def collect(self, obs, value, action, logprob, reward, terminations, truncations):
        pass

    @abc.abstractmethod
    def learn(self, last_value, last_terminations, last_truncations):
        pass