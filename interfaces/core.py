import abc

class Core(abc.ABC):

    @abc.abstractmethod
    def get_latest_value(self, x):
        pass

    @abc.abstractmethod
    def get_action_and_value(self, x, action=None):
        pass

    @abc.abstractmethod
    def unpack_action(packed_action):
        pass


class Context_Collector(abc.ABC):
    
    @abc.abstractmethod
    def append(self, obs, action, reward):
        pass

    @abc.abstractmethod
    def clear(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, index):
        pass

    @abc.abstractmethod
    def make_batch(self, batch_led=True):
        pass
