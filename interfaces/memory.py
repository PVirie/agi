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
    IDLE = 0  # no operation
    CREATE = auto()  # create a new node with write_value and link from current node with edge_1 and edge_2
    WRITE_THEN_MOVE = auto()  # write to current node with write_value and move to the linked node with edge_1 and edge_2
    LINK = auto()  # link from current node to another node with edge_1 and edge_2
    RESET = auto()  # reset the memory of the batch
    

class Graph_Memory(abc.ABC):

    @abc.abstractmethod
    def write_then_move(self, batch_indices, write_value, next_edge):
        pass


    @abc.abstractmethod
    def create(self, batch_indices, write_value):
        pass


    @abc.abstractmethod
    def link(self, batch_indices, edge_1, edge_2):
        pass


    @abc.abstractmethod
    def reset(self, batch_indices):
        pass