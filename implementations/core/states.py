import numpy as np

from interfaces.core import Context_Collector


class State_Sequence(Context_Collector):
    
    def __init__(self, max_history: int, data=None, mask=None):
        self.max_history = max_history
        if data is None:
            self.data = []
        else:
            self.data = data

        if mask is None:
            self.mask = []
        else:
            self.mask = mask


    def append(self, *args, **kwargs):
        """
            all arguments are np arrays of shape (batch_size, -1)
        """
        all_args = []
        for arg in args:
            all_args.append(arg)
        for key in kwargs:
            all_args.append(kwargs[key])

        self.data.append(np.concatenate(all_args, axis=1))
        self.mask.append(1.0)


    def clear(self):
        self.data = []
        self.mask = []


    def mark(self, skip_last=False):
        # assign zeroes to all elements, skipping last if specified
        length = len(self.mask)
        end = length - 1 if skip_last else length
        for i in range(end):
            self.mask[i] = 0.0


    def __getitem__(self, slice):
        # get (start-max_history):stop slice
        start = slice.start if slice.start is not None else 0
        stop = slice.stop if slice.stop is not None else len(self.data)
        start = max(0, start - self.max_history)
        return State_Sequence(self.max_history, self.data[start:stop], self.mask[start:stop])
    

    def make_batch(self, batch_led=True, append_last=False):
        # return tensor of shape (batch_size, len(data), :) if batch_led else (len(data), batch_size, :)
        if len(self.data) == 0:
            return None
        
        if append_last:
            extra = np.zeros_like(self.data[0])
            data_to_stack = self.data + [extra]
        else:
            data_to_stack = self.data
        
        if batch_led:
            data_tensor = np.stack(data_to_stack, axis=1)
        else:
            data_tensor = np.stack(data_to_stack, axis=0)
        return data_tensor
    

    def make_mask(self, batch_led=True, append_last=False):
        # return tensor of shape (batch_size, len(data), :) if batch_led else (len(data), batch_size, :)
        if len(self.mask) == 0:
            return None
        
        if append_last:
            mask_to_stack = self.mask + [0.0]
        else:
            mask_to_stack = self.mask
        
        if batch_led:
            mask_tensor = np.stack(mask_to_stack, axis=1)
        else:
            mask_tensor = np.stack(mask_to_stack, axis=0)
        return mask_tensor