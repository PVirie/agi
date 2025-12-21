import abc
from enum import Enum
from typing import List


class Memory_Operation_Type(Enum):
    CACHE = 0
    FETCH_BY_POSITION = 1
    FETCH_BY_CONTENT = 2
    RESET = 3
    IDLE = 4  # no operation


class Memory(abc.ABC):

    @abc.abstractmethod
    def operate(self, position, content, operations: List[Memory_Operation_Type]):
        pass
