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
        o1_norm = np.linalg.norm(o1, axis=1, keepdims=True) + 1e-8  # (slot_size)
        o2_norm = np.linalg.norm(o2) + 1e-8  # scalar
        o1_normalized = o1 / o1_norm  # (slot_size, dim)
        o2_normalized = o2 / o2_norm  # (dim)
        dot_score = np.matmul(o1_normalized, o2_normalized)  # (slot_size)
        energy = np.exp(dot_score - np.max(dot_score))  # (slot_size), for numerical stability
        energy = energy / np.sum(energy)  # (slot_size)
        return energy
    
    
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


    def reset(self, batch):
        self.data[batch] = [np.zeros((self.max_slot_size, self.sizes[t])) for t in range(len(self.sizes))]


    def operate(self, tuple_record, operation: List[Memory_Operation_Type], index: List[int]=None, replace_all_index: List[bool]=None):
        """
        tuple_record is a tuple of np array of shape (batch_size, tuple_size[t])
        operation is a list of Memory_Operation_Type with length batch_size
        index is a list of int with length batch_size
        """
        batch_size = tuple_record[0].shape[0]
        best_outputs = [t for t in tuple_record]
        replace_all_index = replace_all_index if replace_all_index is not None else [True for _ in range(batch_size)]
        for i in range(batch_size):
            if i >= len(self.data):
                # initialize new slot
                self.data.append([np.zeros((self.max_slot_size, self.sizes[t])) for t in range(len(self.sizes))])

            flag = operation[i]
            
            if Memory_Operation_Type.RESET in flag:
                # reset
                self.data[i] = [np.zeros((self.max_slot_size, self.sizes[t])) for t in range(len(self.sizes))]

            if Memory_Operation_Type.FETCH in flag:
                # fetch
                t_index = index[i]
                current_records = self.data[i] # [(slot_size, tuple_size[t])]
                prob = self.__infer(current_records[t_index], best_outputs[t_index][i])  # (slot_size)
                for t in range(len(self.sizes)):
                    if not replace_all_index[i] and t == t_index:
                        # fetch, only the parts that are not t_index
                        continue
                    best_outputs[t][i, :] = self.__fetch(current_records[t], prob)

            if Memory_Operation_Type.CACHE in flag:
                # cache (append last)
                for t in range(len(self.sizes)):
                    # shift left
                    self.data[i][t][:-1, :] = self.data[i][t][1:, :]
                    # append new
                    self.data[i][t][-1, :] = best_outputs[t][i, :]

        return best_outputs