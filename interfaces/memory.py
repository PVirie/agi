import abc
from enum import Enum
from typing import List


class Memory_Operation_Type(Enum):
    CACHE = 0
    FETCH = 1
    RESET = 2
    IDLE = 3  # no operation
    FETCH_AND_CACHE = 4


class Memory(abc.ABC):

    @abc.abstractmethod
    def operate(self, tuple_record, operation: List[Memory_Operation_Type], index: List[int]=None):
        pass
