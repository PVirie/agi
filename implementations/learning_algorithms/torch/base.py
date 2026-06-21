import torch
from torch import nn
import numpy as np
from typing import Any, List


def convert_list_of_list_of_bool_to_float_tensor(bool_list_of_list: List[List[bool]], device) -> torch.Tensor:
    np_array = np.stack(bool_list_of_list, axis=1)
    return torch.tensor(np_array.astype(np.float32), dtype=torch.float32).to(device)


def convert_np_array_to_float_tensor(np_array: np.ndarray, device) -> torch.Tensor:
    return torch.tensor(np_array, dtype=torch.float32).to(device)


def convert_np_array_to_bool_tensor(np_array: np.ndarray, device) -> torch.Tensor:
    return torch.tensor(np_array.astype(bool), dtype=torch.bool).to(device)


def convert_np_array_to_int_tensor(np_array: np.ndarray, device) -> torch.Tensor:
    return torch.tensor(np_array.astype(int), dtype=torch.int64).to(device)


def convert_list_of_np_array_to_float_tensor(np_array_list: List[np.ndarray], device) -> List[torch.Tensor]:
    combined = np.stack(np_array_list, axis=1)
    return torch.tensor(combined, dtype=torch.float32).to(device)


def convert_list_of_float_to_float_tensor(float_list: List[List[float]], device) -> torch.Tensor:
    before_transpose = torch.tensor(np.array(float_list, dtype=np.float32), dtype=torch.float32).to(device)
    return torch.transpose(before_transpose, 0, 1)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False) -> torch.Tensor:
    masked_tensor = tensor * mask
    total_elements = mask.sum(dim=dim, keepdim=keepdim)
    return masked_tensor.sum(dim=dim, keepdim=keepdim) / (total_elements + 1e-8)


def masked_std(tensor: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False) -> torch.Tensor:
    mean = masked_mean(tensor, mask, dim=dim, keepdim=True)
    variance = masked_mean((tensor - mean) ** 2, mask, dim=dim, keepdim=keepdim)
    return torch.sqrt(variance + 1e-8)