import torch
from torch import nn
import numpy as np
from typing import Any, List


def convert_list_of_bool_to_float_tensor(bool_list: List[bool], device) -> torch.Tensor:
    return torch.tensor([1.0 if b else 0.0 for b in bool_list], dtype=torch.float32).to(device)


def convert_np_array_to_float_tensor(np_array: np.ndarray, device) -> torch.Tensor:
    return torch.tensor(np_array, dtype=torch.float32).to(device)


def convert_list_of_np_array_to_float_tensor(np_array_list: List[np.ndarray], device) -> List[torch.Tensor]:
    before_transpose = [torch.tensor(arr, dtype=torch.float32).to(device) for arr in np_array_list]
    return torch.stack(before_transpose, dim=1)


def convert_list_of_float_to_float_tensor(float_list: List[float], device) -> torch.Tensor:
    before_transpose = torch.tensor(float_list, dtype=torch.float32).to(device)
    return torch.transpose(before_transpose, 0, 1)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False) -> torch.Tensor:
    masked_tensor = tensor * mask
    if dim is None:
        total_elements = mask.numel()
    else:
        total_elements = mask.sum(dim=dim, keepdim=keepdim)
    
    return masked_tensor.sum(dim=dim, keepdim=keepdim) / (total_elements + 1e-8)


def masked_std(tensor: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False) -> torch.Tensor:
    masked_tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim, keepdim=True)
    variance = masked_mean((masked_tensor - mean) ** 2, mask, dim=dim, keepdim=keepdim)
    return torch.sqrt(variance + 1e-8)