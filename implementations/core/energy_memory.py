import numpy as np
from typing import List

from interfaces.memory import Memory, Memory_Operation_Type


class Energy_Memory(Memory):

    def __init__(self, position_size: int, content_size: int, max_slot_size: int):
        self.data = []

        self.position_size = position_size
        self.content_size = content_size
        self.max_slot_size = max_slot_size


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
        # get the max slot
        max_index = np.argmax(prob)
        content = o1[max_index, :]  # (contedimnt_size)
        return content


    def operate(self, position, content, operations: List[Memory_Operation_Type]):
        """
        position has shape (batch_size, position_size)
        content has shape (batch_size, content_size)
        operations is a list of Memory_Operation_Type with length batch_size
        """
        batch_size = position.shape[0]
        best_positions = position
        best_contents = content
        for i in range(batch_size):
            if i >= len(self.data):
                # initialize new slot
                self.data.append({
                    "positions": np.zeros((self.max_slot_size, self.position_size)),
                    "contents": np.zeros((self.max_slot_size, self.content_size)),
                })

            flag = operations[i]
            if flag == Memory_Operation_Type.IDLE:
                # no operation
                continue
            elif flag == Memory_Operation_Type.RESET:
                # reset
                self.data[i] = {
                    "positions": np.zeros((self.max_slot_size, self.position_size)),
                    "contents": np.zeros((self.max_slot_size, self.content_size)),
                }
            elif flag == Memory_Operation_Type.FETCH_BY_POSITION:
                # fetch by position
                slot_positions = self.data[i]["positions"]  # (slot_size, position_size)
                slot_contents = self.data[i]["contents"]    # (slot_size, content_size)

                prob = self.__infer(slot_positions, position[i])  # (slot_size)
                best_position = self.__fetch(slot_positions, prob)  # (position_size)
                best_content = self.__fetch(slot_contents, prob)    # (content_size)

                best_positions[i, :] = best_position
                best_contents[i, :] = best_content
            elif flag == Memory_Operation_Type.FETCH_BY_CONTENT:
                # fetch by content
                slot_positions = self.data[i]["positions"]  # (slot_size, position_size)
                slot_contents = self.data[i]["contents"]    # (slot_size, content_size)

                prob = self.__infer(slot_contents, content[i])  # (slot_size)
                best_position = self.__fetch(slot_positions, prob)  # (position_size)
                best_content = self.__fetch(slot_contents, prob)    # (content_size)

                best_positions[i, :] = best_position
                best_contents[i, :] = best_content
            elif flag == Memory_Operation_Type.CACHE:
                # cache (append last)

                slot_positions = self.data[i]["positions"]  # (slot_size, position_size)
                slot_contents = self.data[i]["contents"]    # (slot_size, content_size)

                # shift left
                slot_positions[:-1, :] = slot_positions[1:, :]
                slot_contents[:-1, :] = slot_contents[1:, :]

                # append new
                slot_positions[-1, :] = position[i, :]
                slot_contents[-1, :] = content[i, :]

                self.data[i]["positions"] = slot_positions
                self.data[i]["contents"] = slot_contents

        return best_positions, best_contents