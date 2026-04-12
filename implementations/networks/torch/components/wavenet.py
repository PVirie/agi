import torch
import torch.nn as nn
import torch.nn.functional as F


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