import numpy as np
from typing import List

from interfaces.memory import Episodic_Memory


class Energy_Episodic_Memory(Episodic_Memory):

    def __init__(self, sizes: List[int], max_slot_size: int):
        self.sizes = sizes
        self.max_slot_size = max_slot_size

        # data is a batch wise list of memory slots 
        # data: List[ Tuple[ np.ndarray of shape (slot_size, tuple_size[t]) ] ]
        self.data = []
        self.current_pointer = []


    def make_batch(self, batch_led=True):
        """
            make a np array of shape (batch_size, slot_size, sum_t(tuple_size[t])) if batch_led is True, 
            otherwise make a np array of shape (slot_size, batch_size, sum_t(tuple_size[t]))
        """
        if batch_led:
            batch_size = len(self.data)
            batch_data = np.zeros((batch_size, self.max_slot_size, sum(self.sizes)), dtype=np.float32)
            for b in range(batch_size):
                for t in range(len(self.sizes)):
                    batch_data[b, :, sum(self.sizes[:t]):sum(self.sizes[:t+1])] = self.data[b][t]
        else:
            batch_size = len(self.data)
            batch_data = np.zeros((self.max_slot_size, batch_size, sum(self.sizes)), dtype=np.float32)
            for b in range(batch_size):
                for t in range(len(self.sizes)):
                    batch_data[:, b, sum(self.sizes[:t]):sum(self.sizes[:t+1])] = self.data[b][t]
        return batch_data


    def reset(self, batch):
        self.data[batch] = [np.zeros((self.max_slot_size, self.sizes[t])) for t in range(len(self.sizes))]
        self.current_pointer[batch] = 0


    def cache(self, batch: int, tuple_record):
        while len(self.data) <= batch:
            self.data.append([np.zeros((self.max_slot_size, self.sizes[t])) for t in range(len(self.sizes))])
            self.current_pointer.append(0)

        pointer = self.current_pointer[batch]
        for t in range(len(self.sizes)):
            self.data[batch][t][pointer] = tuple_record[t]

        self.current_pointer[batch] = (pointer + 1) % self.max_slot_size
        return pointer


    def fetch(self, batch: int, tuple_record, pivot_index=1):
        while len(self.data) <= batch:
            self.data.append([np.zeros((self.max_slot_size, self.sizes[t])) for t in range(len(self.sizes))])
            self.current_pointer.append(0)

        prob = self.__infer(self.data[batch][pivot_index], tuple_record[pivot_index])
        max_index = self.__get_max(prob)
        return max_index


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
    
    
    def __get_max(self, prob):
        """
        prob: (slot_size)
        return: max_index: int
        """
        # get the max slot, if same prob, get the furthest one
        # max_index = np.argmax(prob)
        max_value = np.max(prob)
        candidate_indices = np.where(prob == max_value)[0]
        max_index = candidate_indices[-1]  # furthest one
        return max_index
    

        
