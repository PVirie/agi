import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.init import trunc_normal_
from torch.distributions.categorical import Categorical

import numpy as np


def Sinusoidal_positional_encoding(seq_len, embed_dim):
    pe = torch.zeros(seq_len, embed_dim)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def causal_mask(size, device):
    """
    in pytorch 2.6, mask is a boolean tensor where True means to be masked out.
    """
    # make
    # tensor([
    #     [F, T, T, T],
    #     [F, F, T, T],
    #     [F, F, F, T],
    #     [F, F, F, F]
    # ])
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return mask


def apply_transformer(decoder, input, memory=None, tgt_mask=None, tgt_is_causal=False):
    input = input.permute(1, 0, 2)
    if memory is not None:
        memory = memory.permute(1, 0, 2)
    else:
        memory = input
    output = decoder(input, memory, tgt_mask=tgt_mask, tgt_is_causal=tgt_is_causal) # n_contexts x batch x hidden
    output = output.permute(1, 0, 2) # batch x n_contexts x hidden
    return output


def softmax_with_temperature(logits, temperature=1.0, dim=-1):
    """
    Numerically stable softmax with temperature scaling.

    Args:
        logits (torch.Tensor): The input tensor of logits.
        temperature (float, optional): The temperature parameter. Defaults to 1.0.
        dim (int, optional): The dimension along which to compute softmax. Defaults to -1.

    Returns:
        torch.Tensor: The softmax output with temperature scaling.
    """
    with torch.no_grad():
        max_values = torch.max(logits, dim=dim, keepdim=True)[0]
    shifted_logits = (logits - max_values) / temperature
    
    exp_values = torch.exp(shifted_logits) + 1e-8 # Adding a small constant to avoid numerical issues
    sum_exp_values = torch.sum(exp_values, dim=dim, keepdim=True)
    
    softmax_output = exp_values / sum_exp_values
    
    return softmax_output


def log_softmax_with_temperature(logits, temperature=1.0, dim=-1):
    """
    Numerically stable log softmax with temperature scaling.

    Args:
        logits (torch.Tensor): The input tensor of logits.
        temperature (float, optional): The temperature parameter. Defaults to 1.0.
        dim (int, optional): The dimension along which to compute log softmax. Defaults to -1.

    Returns:
        torch.Tensor: The log softmax output with temperature scaling.
    """
    with torch.no_grad():
        max_values = torch.max(logits, dim=dim, keepdim=True)[0]
    shifted_logits = (logits - max_values) / temperature
    
    log_exp_values = shifted_logits - torch.log(torch.sum(torch.exp(shifted_logits), dim=dim, keepdim=True) + 1e-8)
    
    return log_exp_values


class Log_Softmax_Function(torch.autograd.Function):
    """
    Custom Log Softmax function with manually defined gradient.
    Log Softmax: log(softmax(x)) = x - log(sum(exp(x))).
    Avoids exp in the backward pass for stability or specific numerical reasons.
    """
    MIN_BOUND = -10

    @staticmethod
    def forward(ctx, logits, temperature=1.0, dim=-1):
        ctx.temperature = temperature
        ctx.dim = dim
        max_values = torch.max(logits, dim=dim, keepdim=True)[0]
        shifted_logits = (logits - max_values) / temperature
        exp_values = torch.exp(shifted_logits)
        sum_exp_values = torch.sum(exp_values, dim=dim, keepdim=True).clamp_min(1e-38) # Avoid log(0)
        log_softmax_output = shifted_logits - torch.log(sum_exp_values)
        ctx.save_for_backward(log_softmax_output)  # Save log softmax output for backward pass

        return torch.clamp(log_softmax_output, min=Log_Softmax_Function.MIN_BOUND)


    @staticmethod
    def backward(ctx, grad_output):
        temperature = ctx.temperature
        dim = ctx.dim
        log_softmax_unclamped, = ctx.saved_tensors  # Unpack saved tensor

        clamp_min_val = Log_Softmax_Function.MIN_BOUND
        mask = (log_softmax_unclamped > clamp_min_val).type_as(grad_output)
        adj_grad_output = grad_output * mask
        p = torch.exp(log_softmax_unclamped)  # Compute probabilities from log softmax
        grad_shifted_logits = adj_grad_output - p * torch.sum(adj_grad_output, dim=dim, keepdim=True)
        
        return grad_shifted_logits / temperature, None, None  # No gradient for temperature and dim


class Exp_Entropy_Function(torch.autograd.Function):
    """
    Custom Entropy function with manually defined gradient.
    Entropy: H(log(p)) = - sum(p * log(p)).
    Avoids p * log(p) in the backward pass for stability or specific numerical reasons.
    """
    @staticmethod
    def forward(ctx, log_p, dim):
        p = torch.exp(log_p)
        entropy = -torch.sum(p * log_p, dim=dim)

        # Save p_stable for backward pass (or log_p directly)
        ctx.save_for_backward(log_p)
        ctx.dim = dim
        return entropy


    @staticmethod
    def backward(ctx, grad_output):
        log_p, = ctx.saved_tensors # Retrieve saved tensor
        dim = ctx.dim
        p = torch.clamp(torch.exp(log_p), min=1e-10)
        grad_entropy_p = -p * (1 + log_p)
        grad_input = grad_entropy_p * grad_output.unsqueeze(dim)

        # The 'dim' argument does not require a gradient
        return grad_input, None


class Multilayer_Relu(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers=1, device=None):
        super(Multilayer_Relu, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size, device=device))
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size, device=device))
        self.layers.append(nn.Linear(hidden_size, output_size, device=device))


    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x
    

    def reset_parameters(self):
        self.apply(init_weights)


class Res_Net(Multilayer_Relu):
    def __init__(self, input_size, output_size, hidden_size, n_layers=1, device=None):
        super(Res_Net, self).__init__(input_size, output_size, hidden_size, n_layers, device)


    def forward(self, x):
        for layer in self.layers[:-1]:
            pre_x = x
            x = F.relu(layer(x))
            x = x + pre_x
        x = self.layers[-1](x)
        return x
    

    def reset_parameters(self):
        self.apply(init_weights)


def reset_transformer_decoder(module):
    for decoder_layer in module.modules():
        for inner_module in decoder_layer.modules():
            if hasattr(inner_module, 'reset_parameters'):
                inner_module.reset_parameters()
            elif hasattr(inner_module, '_reset_parameters'):
                inner_module._reset_parameters()


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)



class Multilayer_CNN(nn.Module):
    def __init__(self, input_channels, output_size, hidden_channels, n_layers=1, kernel_size=3, device=None):
        super(Multilayer_CNN, self).__init__()
        self.device = device
        self.hidden_channels = hidden_channels
        self.layers = nn.Sequential()
        # add cnn blocks with group norm and relu
        for i in range(n_layers):
            in_channels = input_channels if i == 0 else hidden_channels
            self.layers.add_module(f'conv_{i}', nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size//2, device=device))
            self.layers.add_module(f'groupnorm_{i}', nn.GroupNorm(num_groups=1, num_channels=hidden_channels, device=device))
            self.layers.add_module(f'relu_{i}', nn.ReLU())
        # final fc layer
        self.fc = nn.Linear(hidden_channels, output_size, device=device)

    # output shape: (batch, output_size)
    def forward(self, x):
        x = self.layers(x)
        # global average pooling
        x = torch.mean(x, dim=(2, 3))
        x = self.fc(x)
        return x
    

    def reset_parameters(self):
        self.apply(init_weights)


# Implement pytorch Categorial with mask sample (Unlike MaskedCategorical)
class Categorical_With_Mask_Sample(Categorical):
    def masked_sample(self, mask=None, sample_shape=torch.Size()):
        """
        Samples with a mask, handling automatic broadcasting for missing dimensions.
        
        Broadcasting Logic:
        If mask.ndim < logits.ndim, we assume the mask applies to the 
        Leading Dimensions (Batch) and the Last Dimension (Category). 
        We unsqueeze the middle dimensions of the mask to match logits.
        
        Example:
            Logits: (Batch, Sequence, Category)
            Mask:   (Batch, Category)
            -> Mask becomes (Batch, 1, Category) to broadcast over Sequence.
        """
        if mask is None:
            return super().sample(sample_shape)
            
        # 1. Align Mask Dimensions (Middle Broadcasting)
        if mask.ndim < self.logits.ndim:
            # Calculate how many middle dimensions are missing (e.g., Sequence)
            # Logits: (B, S, C) [ndim=3] | Mask: (B, C) [ndim=2] -> diff = 1
            diff = self.logits.ndim - mask.ndim
            
            # Reshape mask: Keep leading dims, insert 1s, keep last dim
            # (B, C) -> (B, 1, C)
            new_shape = mask.shape[:-1] + (1,) * diff + mask.shape[-1:]
            mask = mask.view(new_shape)

        # 2. Safety Check
        # Ensure that AFTER broadcasting, we don't have any completely blocked rows
        # We broadcast the mask against logits shape (excluding the last dim) to check validity
        # Note: We use expand_as to simulate the broadcast without copying data
        mask_expanded = mask.expand_as(self.logits)
        if not mask_expanded.any(dim=-1).all():
             raise ValueError("Invalid mask: at least one batch/sequence item has NO valid actions.")

        # 3. Apply Mask
        logits = self.logits.clone()
        logits[~mask_expanded.bool()] = -float('inf')
        
        # 4. Sample
        return Categorical(logits=logits).sample(sample_shape)

    def sample_from_available_indices(self, indices=None, sample_shape=torch.Size()):
        """
        Creates a minimal mask from indices and delegates to masked_sample
        to handle the broadcasting logic.
        """
        if indices is None:
            return super().sample(sample_shape)

        # Initialize minimal mask based on indices shape, NOT logits shape.
        # We let masked_sample handle the expansion to match logits later.
        
        # Case A: Ragged Lists [[0,1], [2]] -> Mask (Batch, Category)
        if isinstance(indices, (list, tuple)):
            # Determine shape: (Batch_Size, Category_Size)
            batch_size = len(indices)
            cat_size = self.logits.shape[-1]
            
            # Create base mask on correct device
            mask = torch.zeros((batch_size, cat_size), device=self.logits.device, dtype=torch.bool)
            
            for i, valid_idx in enumerate(indices):
                if hasattr(valid_idx, '__len__') and len(valid_idx) > 0:
                    idx_tensor = torch.as_tensor(valid_idx, device=self.logits.device, dtype=torch.long)
                    mask[i, idx_tensor] = True
                elif isinstance(valid_idx, int): # Handle simple list [0, 1]
                     mask[i, valid_idx] = True

        # Case B: Tensor Indices
        elif isinstance(indices, torch.Tensor):
            indices = indices.to(self.logits.device).long()
            
            # Create mask with same rank as indices (plus category dim)
            # If indices is (B, K), mask becomes (B, C)
            mask_shape = indices.shape[:-1] + (self.logits.shape[-1],)
            mask = torch.zeros(mask_shape, device=self.logits.device, dtype=torch.bool)
            
            # Scatter True values
            # value=1 broadcasts to the shape of indices
            mask.scatter_(dim=-1, index=indices, value=True)
            
        else:
            raise ValueError("indices must be None, List, or Tensor")

        return self.masked_sample(mask=mask, sample_shape=sample_shape)