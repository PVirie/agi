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
    

def gumbel_rao_straight_through(logits, index, tau=1.0, num_samples=1, mask=None):
    """
    Straight-through Gumbel-Rao one-hot (Paulus et al., 2021) of a category that was already drawn.
    Forward: exactly one_hot(index). Backward: the gradient of E[softmax((logits + G) / tau) | argmax(logits + G) = index],
    estimated with num_samples Gumbel draws conditioned on that argmax. Conditioning on the drawn index (instead of
    reusing its noise, as plain ST Gumbel-softmax would) lets the relaxation be rebuilt for a replayed rollout action.

    logits: (..., K); index: (...) integer; mask: (..., K) bool, False marks an impossible category.
    Returns: (..., K)
    """
    if mask is not None:
        # finite (not -inf) so the truncated-Gumbel arithmetic below never meets inf - inf
        logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
    index = index.long()
    hard = F.one_hot(index, num_classes=logits.size(-1)).to(logits.dtype)

    # the noise is a constant, as in ST Gumbel-softmax: only its distribution is conditioned on index
    with torch.no_grad():
        detached = logits.detach()
        tiny = torch.finfo(logits.dtype).tiny
        uniform = torch.rand((num_samples,) + tuple(logits.shape), device=logits.device, dtype=logits.dtype)
        gumbels = -torch.log(-torch.log(uniform.clamp(min=tiny, max=1.0 - 1e-7)))
        # the max of the perturbed logits is Gumbel(logsumexp(logits)) and sits at index
        top = torch.logsumexp(detached, dim=-1, keepdim=True) + gumbels.gather(-1, index.unsqueeze(-1).expand((num_samples,) + tuple(index.shape) + (1,)))
        # every other perturbed logit is Gumbel(logit) truncated below that max
        others = -torch.logaddexp(-top, -(detached + gumbels))
        noise = torch.where(hard.bool(), top, others) - detached

    soft = torch.softmax((logits + noise) / tau, dim=-1).mean(dim=0)
    # soft - soft.detach() is exactly 0 in the forward pass, so the output equals the hard one-hot bit for bit
    return hard + (soft - soft.detach())


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

if __name__ == "__main__":
    torch.manual_seed(0)

    logits = torch.tensor([0.5, -1.0, 2.0, 0.0, 1.0], requires_grad=True)
    index = torch.tensor([2, 0, 4])
    out = gumbel_rao_straight_through(logits.expand(3, -1), index, num_samples=4)
    assert torch.equal(out, F.one_hot(index, 5).float()), "forward must be the exact one-hot"

    # Rao-Blackwellization keeps the ST Gumbel-softmax gradient in expectation
    n, tau = 1000000, 0.7
    weights = torch.randn(5)
    drawn = Categorical(logits=logits.detach()).sample((n,))
    rao_grad, = torch.autograd.grad((gumbel_rao_straight_through(logits.expand(n, -1), drawn, tau=tau) * weights).sum() / n, logits)
    gumbels = -torch.log(-torch.log(torch.rand(n, 5)))
    st_grad, = torch.autograd.grad((torch.softmax((logits + gumbels) / tau, dim=-1) * weights).sum() / n, logits)
    assert torch.allclose(rao_grad, st_grad, atol=0.02), f"Gumbel-Rao gradient {rao_grad} != ST Gumbel-softmax gradient {st_grad}"

    mask = torch.tensor([True, False, True, True, False])
    masked_grad, = torch.autograd.grad((gumbel_rao_straight_through(logits.expand(64, -1), torch.full((64,), 2), mask=mask.expand(64, -1)) * weights).sum(), logits)
    assert torch.isfinite(masked_grad).all() and (masked_grad[~mask] == 0).all(), f"masked categories must get no gradient: {masked_grad}"

    print("All tests passed!")
