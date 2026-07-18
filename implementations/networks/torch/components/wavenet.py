import torch
import torch.nn as nn
import torch.nn.functional as F

from implementations.networks.torch.components.base import init_weights


class BasicResidualBlock(nn.Module):
    """Causal residual block with left zero-padding to preserve sequence length.
    kernel_size=2: output at t uses inputs [t-dilation, t]."""
    def __init__(self, d_model, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=2, dilation=dilation, padding=0)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.activation = nn.ReLU()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        out = self.activation(x)
        out = F.pad(out, (self.dilation, 0))  # left-only pad to keep length
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return x + out


class BasicWavenet(nn.Module):
    """Causal WaveNet (kernel_size=2, original style) with left zero-padding.

    Dilations double each layer: 1, 2, 4, ..., 2^(num_layers-1).
    Total causal receptive field = 2^num_layers positions.
    Input and output are both (B, L, D).

    Args:
        d_model: channel dimension.
        num_layers: number of dilated residual blocks.
    """
    def __init__(self, d_model=64, num_layers=10):
        super().__init__()
        self.residual_blocks = nn.ModuleList([
            BasicResidualBlock(d_model, 2 ** i)
            for i in range(num_layers)
        ])

    def reset_parameters(self):
        for block in self.residual_blocks:
            block.reset_parameters()

    def forward(self, x):
        # x: (B, L, D)
        x = x.transpose(1, 2)   # (B, D, L)
        for block in self.residual_blocks:
            x = block(x)
        x = x.transpose(1, 2)   # (B, L, D)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, d_model, kernel_size, dilation):
        super().__init__()
        self.pad = dilation * (kernel_size - 1)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, dilation=dilation, padding=0)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = F.pad(x, (self.pad, 0))  # Pad only on the left for causality
        out = self.activation(out)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return x + out  # Residual connection


class Wavenet(nn.Module):
    def __init__(self, d_model=64, num_layers=10, history_steps=None):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        if history_steps is None:
            history_steps = 2
        
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.residual_blocks.append(ResidualBlock(d_model, 1 + history_steps, dilation))


    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x.transpose(1, 2)   # (batch_size, d_model, seq_len)
        for block in self.residual_blocks:
            x = block(x)
        x = x.transpose(1, 2)   # (batch_size, seq_len, d_model)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Decoder-Only Model
    model = Wavenet(history_steps=5).to(device)
    
    # test whether relative position embeddings do not change output values at different positions
    history_steps = 2
    # 0 < history_steps < inf only work for one layer, because higher layer expand the receptive field
    model_test = Wavenet(d_model=2, history_steps=history_steps, num_layers=1).to(device)
    # stop undeterminism by setting model to eval
    model_test.eval()
    data = torch.randn(1, 4 + history_steps * 2, 2).to(device)
    shifted_data = torch.roll(data, shifts=history_steps, dims=1)
    out1 = model_test(data)[:, history_steps:(data.shape[1]-history_steps), :]
    out2 = model_test(shifted_data)[:, (2*history_steps):, :]
    assert torch.allclose(out1, out2, atol=1e-4)
    print("Relative position embeddings test successful.")


    # test whether model not violate causality
    model_causality = Wavenet(d_model=16, history_steps=None, num_layers=1).to(device)
    model_causality.eval()
    seq_len = 10
    data_causality = torch.randn(1, seq_len, 16).to(device)
    out_causality = model_causality(data_causality)
    for t in range(1, seq_len):
        # output at position t should not depend on input at position > t
        input_modified = data_causality.clone()
        input_modified[0, t+1:, :] += 10
        out_modified = model_causality(input_modified)
        assert torch.allclose(out_causality[0, t, :], out_modified[0, t, :], atol=1e-4)
    print("Causality test successful.")

    # test whether model violates history constraint
    history_limit = 1
    model_history = Wavenet(d_model=16, history_steps=history_limit, num_layers=1).to(device)
    model_history.eval()
    seq_len = 10
    data_history = torch.randn(1, seq_len, 16).to(device)
    out_history = model_history(data_history)
    for t in range(seq_len):
        # output at position t should not depend on input at position < t - history_limit
        input_modified = data_history.clone()
        if t - history_limit - 1 >= 0:
            input_modified[0, :t - history_limit - 1, :] += 10
            out_modified = model_history(input_modified)
            assert torch.allclose(out_history[0, t, :], out_modified[0, t, :], atol=1e-4)
    print("History constraint test successful.")

    # test BasicWavenet input/output shape
    for batch, seq_len, d_model, num_layers in [(1, 16, 32, 4), (4, 100, 64, 10), (2, 1, 8, 3)]:
        model_basic = BasicWavenet(d_model=d_model, num_layers=num_layers).to(device)
        model_basic.eval()
        x = torch.randn(batch, seq_len, d_model).to(device)
        y = model_basic(x)
        assert y.shape == x.shape, f"Shape mismatch: input {x.shape} vs output {y.shape}"
    print("BasicWavenet shape test successful.")