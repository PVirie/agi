import abc
from enum import Flag, auto
from typing import List


class Memory_Operation_Type(Flag):
    IDLE = 0  # no operation
    RESET = auto()
    FETCH = auto()
    CACHE = auto()


class Memory(abc.ABC):

    @abc.abstractmethod
    def operate(self, tuple_record, operation: List[Memory_Operation_Type], index: List[int]=None, replace_all_index: List[bool]=None):
        pass
