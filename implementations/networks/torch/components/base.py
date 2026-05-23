import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.init import trunc_normal_
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import Bernoulli

import numpy as np


def init_weights(m):
    if isinstance(m, nn.Linear):
        # Orthogonal init is generally better for RL (PPO/A2C)
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.GroupNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.InstanceNorm2d):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# Implement pytorch Categorial with mask (Same as MaskedCategorical)
class Categorical_With_Mask(Categorical):
    
    def __init__(self, probs=None, logits=None, mask=None, validate_args=None):
        """
        Args:
            probs (Tensor): Event probabilities. 
            logits (Tensor): Event log-probabilities.
            mask (Tensor): Boolean tensor where False indicates the category 
                           should be masked out (probability = 0).
            validate_args (bool): Whether to validate input values.
        """
        self.mask = mask
        
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")

        if mask is not None:
            if probs is not None:
                logits = torch.log(probs)
                probs = None # Ensure we only pass logits to super()
            min_value = torch.finfo(logits.dtype).min
            logits = torch.where(mask, logits, torch.tensor(min_value).to(logits.device))

        super().__init__(probs=probs, logits=logits, validate_args=validate_args)


    def expand(self, batch_shape, _instance=None):
        """
        Expand implementation to ensure the mask is carried over correctly 
        when .expand() is called on the distribution.
        """
        new = super().expand(batch_shape, _instance)
        
        if self.mask is not None:
            try:
                new.mask = self.mask.expand(batch_shape + self.event_shape)
            except RuntimeError:
                new.mask = self.mask
        return new
    

def batched_backward_fill_with_mask(flag: torch.Tensor):
    """
    Backward fills the array positions with the index of the next '1'.
    For example [0, 1, 0, 1, 0, 1] -> [1, 1, 3, 3, 5, 5]
    Supports arbitrary batch dimensions, e.g., (B, L) or (B, H, L).
    
    Args:
        flag (torch.Tensor): Tensor of 0s and 1s with shape (*, L).
        
    Returns:
        backward_filled (torch.Tensor): Tensor with backward filled indices, shape (*, L).
        mask (torch.Tensor): Boolean mask that is True where no subsequent '1' was found.
    """
    # The sequence length is the size of the last dimension
    L = flag.size(-1)
    
    # 1. Create 1D array of indices: [0, 1, ..., L-1]
    # PyTorch will automatically broadcast this to match the batch dimensions of 'flag'
    indices = torch.arange(L, device=flag.device)
    
    # 2. Replace positions where flag is 0 with L (acts as infinity)
    vals = torch.where(flag == 1, indices, L)
    
    # 3. Flip, cummin, and flip back along the last dimension (dim=-1)
    backward_filled = vals.flip(-1).cummin(-1).values.flip(-1)
    
    # 4. Create the mask for edge cases
    mask = (backward_filled == L)
    
    # Replace the out-of-bounds placeholders with 0
    backward_filled = torch.where(mask, torch.tensor(0, device=flag.device), backward_filled)

    # revert mask to valid_mask
    mask = ~mask
    
    return backward_filled, mask