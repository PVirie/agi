import abc

class Context_Collector(abc.ABC):
    
    @abc.abstractmethod
    def append(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def update_last(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_last(self):
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