import numpy as np

from interfaces.data_structure import Context_Collector


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

        self.data[-1] = np.concatenate(all_args, axis=1)


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
        # WARNING: if you do this [x:y], the returned sequence will be [x - max_history : y] !!!
        start = slice.start if slice.start is not None else 0
        stop = slice.stop if slice.stop is not None else len(self.data)
        start = max(0, start - self.max_history)
        return State_Sequence(self.max_history, self.data[start:stop], self.mask[start:stop])
    

    def get_last_batch(self, batch_led=True):
        # return tensor of shape (batch_size, seq_len, :) if batch_led else (seq_len, batch_size, :)
        # seq_len = max_history + 1
        if len(self.data) == 0:
            return None
        
        start_index = max(0, len(self.data) - self.max_history - 1)
        data_to_stack = self.data[start_index:]

        if batch_led:
            data_tensor = np.stack(data_to_stack, axis=1)
        else:
            data_tensor = np.stack(data_to_stack, axis=0)
        return data_tensor


    def make_batch(self, batch_led=True):
        # return tensor of shape (batch_size, len(data), :) if batch_led else (len(data), batch_size, :)
        if len(self.data) == 0:
            return None
        
        data_to_stack = self.data
        
        if batch_led:
            data_tensor = np.stack(data_to_stack, axis=1)
        else:
            data_tensor = np.stack(data_to_stack, axis=0)
        return data_tensor
    

    def make_mask(self, batch_led=True):
        # return tensor of shape (batch_size, len(data)) if batch_led else (len(data), batch_size)
        if len(self.mask) == 0:
            return None
        
        batch_size = self.data[0].shape[0]

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


if __name__ == "__main__":
    # test assert that State_Sequence works as expected
    seq = State_Sequence(max_history=5)
    for i in range(10):
        seq.append(np.zeros((2, 3)) + i, np.zeros((2, 2)) + i)
    batch = seq.make_batch(batch_led=True)
    assert batch.shape == (2, 10, 5)
    mask = seq.make_mask(batch_led=True)
    assert mask.shape == (2, 10)
    seq2 = seq[7:8]
    batch2 = seq2.make_batch(batch_led=True)
    assert batch2.shape == (2, 6, 5)
    mark_slice = seq.mark(skip_last=True)
    assert mark_slice == slice(4, 10)
    batch3 = seq.make_batch(batch_led=True)
    assert batch3.shape == (2, 6, 5)
    batch4 = seq.get_last_batch(batch_led=True)
    assert batch4.shape == (2, 6, 5)


    # test zero max_history
    seq = State_Sequence(max_history=0)
    for i in range(10):
        seq.append(np.zeros((2, 3)) + i, np.zeros((2, 2)) + i)
    seq2 = seq[7:8]
    batch2 = seq2.make_batch(batch_led=True)
    assert batch2.shape == (2, 1, 5)
    batch3 = seq.get_last_batch(batch_led=True)
    assert batch3.shape == (2, 1, 5)