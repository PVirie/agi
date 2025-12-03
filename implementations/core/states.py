import numpy as np

from interfaces.core import Context_Collector


class State_Sequence(Context_Collector):
    
    def __init__(self, max_history: int, data=None):
        self.max_history = max_history
        if data is None:
            self.data = []
        else:
            self.data = data


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
        return State_Sequence(self.max_history, self.data[start:stop])
    

    def make_batch(self, batch_led=True):
        # return tensor of shape (batch_size, len(data), :) if batch_led else (len(data), batch_size, :)
        if len(self.data) == 0:
            return None
        
        if batch_led:
            data_tensor = np.stack(self.data, axis=1)
        else:
            data_tensor = np.stack(self.data, axis=0)
        return data_tensor