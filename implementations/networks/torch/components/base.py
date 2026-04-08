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