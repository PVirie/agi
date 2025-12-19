import abc


class Memory(abc.ABC):

    @abc.abstractmethod
    def reset(self, flags):
        pass

    @abc.abstractmethod
    def cache(self, position, content):
        pass

    @abc.abstractmethod
    def fetch_by_position(self, position):
        pass

    @abc.abstractmethod
    def fetch_by_content(self, content):
        pass