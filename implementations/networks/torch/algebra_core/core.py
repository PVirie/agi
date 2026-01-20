import torch
import torch.nn as nn
import torch.nn.functional as F

from implementations.networks.torch.components.base import init_weights


class KernelBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(KernelBlock, self).__init__()

        # every time kernel apply, the dimension is reduced by 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        
        self.res1 = self._build_res_pair(out_channels)
        self.res2 = self._build_res_pair(out_channels)


    def _build_res_pair(self, channels):
        return nn.Sequential(
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x + self.res1(x)
        x = x + self.res2(x)
        return x
    

class Algebra_Core(nn.Module):
    def __init__(self, position_output_dim, feature_dim, num_algebras, context_size):
        super(Algebra_Core, self).__init__()

        self.F = feature_dim
        self.P = position_output_dim
        self.A = num_algebras
        self.C = context_size

        self.num_layers = (context_size - 1) // 2
        self.kernels = nn.Sequential()
        for i in range(self.num_layers):
            in_channels = 1 if i == 0 else self.A
            out_channels = self.A
            self.kernels.add_module(f"kernel_block_{i}", KernelBlock(in_channels, out_channels))

        self.position_kernels = nn.Sequential()
        for i in range(self.num_layers):
            in_channels = 1 if i == 0 else self.P * self.A
            out_channels = self.P * self.A
            self.position_kernels.add_module(f"position_kernel_block_{i}", KernelBlock(in_channels, out_channels))
        

    def forward(self, x):
        """
        :param x: (B, T, F)
        :return: (B, T, P)
        """
        B = x.shape[0]
        T = x.shape[1]

        # perform cosine similarity computation along the feature dimension
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x_transpose = x.permute(0, 1, 3, 2)  # (B, 1, F, T)
        x = torch.matmul(x, x_transpose) / self.F  # (B, 1, T, T)

        # apply kernels
        m = self.kernels(x)  # (B, A, T-C+1, T-C+1)
        p = self.position_kernels(x) # (B, P*A, T-C+1, T-C+1)

        # extract center diagonal
        m = torch.diagonal(m, dim1=2, dim2=3).permute(0, 2, 1).contiguous()  # (B, T-C+1, A)
        p = torch.diagonal(p, dim1=2, dim2=3).permute(0, 2, 1).contiguous()  # (B, T-C+1, P*A)

        # compute softmax over algebra dimension
        m = F.softmax(m, dim=-1)  # (B, T-C+1, A)

        # compute weighted sum over algebra dimension to get position output
        p = p.view(B, T - self.C + 1, self.P, self.A)  # (B, T-C+1, P, A)
        out = torch.einsum('btpa, bta -> btp', p, m)  # (B, T-C+1, P)

        # left pad to (B, T, P)
        pad_size = self.C - 1
        out = F.pad(out, (0, 0, pad_size, 0), mode='constant', value=0.0)  # (B, T, P)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    context_size = 5
    history_steps = 10

    model = Algebra_Core(position_output_dim=2, feature_dim=4, num_algebras=12, context_size=context_size).to(device)
    model.apply(init_weights)
    model.eval()

    # test whether the model does not change output values at different positions
    data = torch.randn(1, context_size + history_steps * 2, 16).to(device)
    shifted_data = torch.roll(data, shifts=history_steps, dims=1)
    out1 = model(data)[:, history_steps:(data.shape[1]-history_steps), :]
    out2 = model(shifted_data)[:, (2*history_steps):, :]
    assert torch.allclose(out1, out2, atol=1e-4)
    print("Relative position embeddings test successful.")


    # test whether model not violate causality
    data_causality = torch.randn(1, context_size + history_steps, 16).to(device)
    out_causality = model(data_causality)  # (1, context_size + history_steps, p)
    for t in range(context_size + history_steps):
        # output at position t should not depend on input at position > t
        input_modified = data_causality.clone()
        input_modified[0, t+1:, :] += torch.randn_like(input_modified[0, t+1:, :])  # some change
        out_modified = model(input_modified)
        assert torch.allclose(out_causality[0, t, :], out_modified[0, t, :], atol=1e-4)
    print("Causality test successful.")