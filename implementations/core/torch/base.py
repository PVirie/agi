import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


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
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()


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
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()


def reset_transformer_decoder(module):
    for decoder_layer in module.modules():
        for inner_module in decoder_layer.modules():
            if hasattr(inner_module, 'reset_parameters'):
                inner_module.reset_parameters()
            elif hasattr(inner_module, '_reset_parameters'):
                inner_module._reset_parameters()