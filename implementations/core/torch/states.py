import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from interfaces.core import Context_Collector


class State_Sequence(Context_Collector):
    
    def __init__(self, position_size: int, max_history: int, data=None, device='cpu'):
        self.position_size = position_size
        self.max_history = max_history
        self.device = device
        if data is None:
            self.data = []
        else:
            self.data = data


    def append(self, reward, position, content):
        """
            reward: np array of shape (batch_size)
            position: tensor of shape (batch_size, position_size) | None
            content: tensor of shape (batch_size, content_size) | np array of shape (batch_size, content_size)
        """
        if isinstance(content, np.ndarray):
            content = torch.tensor(content, dtype=torch.float32).to(self.device)

        if isinstance(position, np.ndarray):
            position = torch.tensor(position, dtype=torch.float32).to(self.device)
        elif position is None:
            position = torch.zeros((content.size(0), self.position_size), dtype=torch.float32).to(self.device)

        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(-1).to(self.device)
        
        self.data.append(torch.cat([reward, position, content], dim=-1))


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
    

    def make_batch(self, batch_led=True):
        # return tensor of shape (batch_size, len(data), :) if batch_led else (len(data), batch_size, :)
        if len(self.data) == 0:
            return None
        data_tensor = torch.stack(self.data, dim=0).to(self.device)
        if not batch_led:
            data_tensor = data_tensor.transpose(0, 1)
        return data_tensor