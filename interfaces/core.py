import abc

class Core(abc.ABC):

    @abc.abstractmethod
    def get_latest_value(self, context, action):
        pass

    @abc.abstractmethod
    def get_action_and_value(self, context, action, use_action=False, use_grad=True):
        pass

    @abc.abstractmethod
    def unpack_action(self, packed_action):
        pass


class Context_Collector(abc.ABC):
    
    @abc.abstractmethod
    def append(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def clear(self):
        pass

    @abc.abstractmethod
    def mark(self, skip_last=False) -> slice:
        pass

    @abc.abstractmethod
    def __getitem__(self, index):
        pass

    @abc.abstractmethod
    def make_batch(self, batch_led=True, append_last=False):
        pass

    @abc.abstractmethod
    def make_mask(self, batch_led=True, append_last=False):
        pass