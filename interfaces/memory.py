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


class Graph_Memory_Operation_Type(Flag):
    IDLE = 0
    RESET = auto()
    CREATE = auto()
    WRITE = auto()
    MOVE = auto()
    LINK = auto()
    ROTATE = auto()


class Graph_Memory:
    def write(self, batch_indices, write_value):
        pass

    def move(self, batch_indices, next_edge):
        pass

    def create(self, batch_indices, write_value):
        pass

    def link(self, batch_indices, edge_1, edge_2):
        pass

    def rotate(self, batch_indices, edge_1, edge_2):
        pass

    def reset(self, batch_indices):
        pass

    def get_edge_update_time(self):
        pass

    def reset_timestamp(self):
        pass