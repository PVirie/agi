import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from interfaces.core import Context_Collector


class State_Sequence(Context_Collector):
    
    def __init__(self, max_history: int, data=None, device='cpu'):
        self.max_history = max_history
        self.device = device
        if data is None:
            self.data = []
        else:
            self.data = data


    def append(self, obs, action, reward):
        self.data.append(torch.concat([obs.flatten(), action.flatten(), torch.tensor([reward], device=obs.device)]))


    def clear(self):
        self.data = []


    def reset(self):
        if len(self.data) > self.max_history:
            self.data = self.data[-self.max_history:]


    def __getitem__(self, slice):
        # get (start-max_history):stop slice
        start = slice.start if slice.start is not None else 0
        stop = slice.stop if slice.stop is not None else len(self.data)
        start = max(0, start - self.max_history)
        return State_Sequence(self.max_history, self.data[start:stop], device=self.device)