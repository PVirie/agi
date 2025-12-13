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


    def update_last(self, *args, **kwargs):
        """
            all arguments are np arrays of shape (batch_size, -1)
        """
        all_args = []
        for arg in args:
            all_args.append(arg)
        for key in kwargs:
            all_args.append(kwargs[key])

        self.data[-1] =  np.concatenate(all_args, axis=1)


    def get_last(self):
        return self.data[-1]


    def clear(self):
        self.data = []
        self.mask = []


    def mark(self, skip_last=False) -> slice:
        """
        Clear memory (data and mask), only keep upto last max_history items 
        If skip_last is True, only keep upto last max_history + 1 items

        set the current mask elements to zero, except the last one if skip_last is True

        :return: the slice representing the left over data after marking
        :rtype: slice[Any, Any, Any]
        """
        end_index = len(self.data)
        if skip_last:
            start_index = max(0, len(self.data) - self.max_history - 1)
            self.data = self.data[start_index:]
            self.mask = [0.0] * (len(self.data) - 1) + [1.0]
        else:
            start_index = max(0, len(self.data) - self.max_history)
            self.data = self.data[start_index:]
            self.mask = [0.0] * len(self.data)
        
        return slice(start_index, end_index)
        

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
        
        batch_size = self.data[0].shape[0]

        if append_last:
            mask_to_stack = self.mask + [0.0]
        else:
            mask_to_stack = self.mask
        
        mask_tensor = np.stack(mask_to_stack, axis=0)
        # expand dims to match data shape
        if batch_led:
            mask_tensor = np.expand_dims(mask_tensor, axis=0)
            mask_tensor = np.repeat(mask_tensor, batch_size, axis=0)
        else:
            mask_tensor = np.expand_dims(mask_tensor, axis=1)
            mask_tensor = np.repeat(mask_tensor, batch_size, axis=1)

        return mask_tensor