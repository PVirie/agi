import numpy as np
from typing import List

from interfaces.memory import Memory, Memory_Operation_Type


class Energy_Memory(Memory):

    def __init__(self, sizes: List[int], max_slot_size: int):
        self.sizes = sizes
        self.max_slot_size = max_slot_size

        # data is a batch wise list of memory slots 
        # data: List[ Tuple[ np.ndarray of shape (slot_size, tuple_size[t]) ] ]
        self.data = []


    def __infer(self, o1, o2):
        """
        compute softmax energy between o1 and o2
        o1: (slot_size, dim)
        o2: (dim)
        return: (slot_size)
        """
        dot_score = np.matmul(o1, np.expand_dims(o2, axis=1))  # (slot_size, 1)
        energy = np.exp(dot_score - np.max(dot_score))  # (slot_size, 1)
        energy = energy / np.sum(energy)  # (slot_size, 1)
        return energy[:, 0]  # (slot_size)
    
    
    def __fetch(self, o1, prob):
        """
        o1: (slot_size, dim)
        prob: (slot_size)
        return: content: (dim)
        """
        # get the max slot, if same prob, get the furthest one
        # max_index = np.argmax(prob)
        max_value = np.max(prob)
        candidate_indices = np.where(prob == max_value)[0]
        max_index = candidate_indices[-1]  # furthest one
        content = o1[max_index, :]  # (contedimnt_size)
        return content


    def operate(self, tuple_record, operation: List[Memory_Operation_Type], index: List[int]=None):
        """
        tuple_record is a tuple of np array of shape (batch_size, tuple_size[t])
        operation is a list of Memory_Operation_Type with length batch_size
        index is a list of int with length batch_size
        """
        batch_size = tuple_record[0].shape[0]
        best_outputs = [t for t in tuple_record]
        for i in range(batch_size):
            if i >= len(self.data):
                # initialize new slot
                self.data.append([np.zeros((self.max_slot_size, self.sizes[t])) for t in range(len(self.sizes))])

            flag = operation[i]
            if flag == Memory_Operation_Type.IDLE:
                # no operation
                continue

            elif flag == Memory_Operation_Type.RESET:
                # reset
                self.data[i] = [np.zeros((self.max_slot_size, self.sizes[t])) for t in range(len(self.sizes))]

            elif flag == Memory_Operation_Type.FETCH:
                # fetch
                t_index = index[i]
                current_records = self.data[i] # [(slot_size, tuple_size[t])]
                prob = self.__infer(current_records[t_index], best_outputs[t_index][i])  # (slot_size)
                for t in range(len(self.sizes)):
                    best_outputs[t][i, :] = self.__fetch(current_records[t], prob)

            elif flag == Memory_Operation_Type.CACHE:
                # cache (append last)
                for t in range(len(self.sizes)):
                    # shift left
                    self.data[i][t][:-1, :] = self.data[i][t][1:, :]
                    # append new
                    self.data[i][t][-1, :] = best_outputs[t][i, :]

            elif flag == Memory_Operation_Type.FETCH_AND_CACHE:
                # fetch and cache
                t_index = index[i]
                current_records = self.data[i] # [(slot_size, tuple_size[t])]
                prob = self.__infer(current_records[t_index], best_outputs[t_index][i])  # (slot_size)
                for t in range(len(self.sizes)):
                    # fetch, only the parts that are not t_index
                    if t != t_index:
                        best_outputs[t][i, :] = self.__fetch(current_records[t], prob)
                # now cache with the updated best_outputs
                for t in range(len(self.sizes)):
                    # shift left
                    self.data[i][t][:-1, :] = self.data[i][t][1:, :]
                    # append new
                    self.data[i][t][-1, :] = best_outputs[t][i, :]

        return best_outputs