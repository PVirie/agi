import torch
import torch.nn as nn
import torch.nn.functional as F

from implementations.networks.torch.components.base import init_weights
from implementations.networks.torch.components.std_conv import ImpalaCNN
    

class Algebra_Core(nn.Module):
    def __init__(self, position_output_dim, num_algebras, context_size):
        super(Algebra_Core, self).__init__()

        self.P = position_output_dim
        self.A = num_algebras
        self.C = context_size

        self.num_layers = (context_size - 1) // 2
        self.kernels = ImpalaCNN(output_dims=self.A, input_channels=1, width=self.C, height=self.C, depths=[16, 32, 32])
        self.position_kernels = ImpalaCNN(output_dims=self.P * self.A, input_channels=1, width=self.C, height=self.C, depths=[16, 32, 32])
        

    def forward(self, x):
        """
        :param x: (B, T, F)
        :return: (B, T, P)
        """
        B = x.shape[0]
        T = x.shape[1]

        # First pad the x along dim 1
        x = F.pad(x, (0, 0, self.C - 1, 0), mode='constant', value=0)  # (B, T + C - 1, F)

        # perform cosine similarity computation along the feature dimension

        # Instead first fold x into smaller (B*T, F, C) tensor
        x_folded = x.unfold(dimension=1, size=self.C, step=1)  # (B, T, F, C)
        x_folded = x_folded.contiguous().view(B * T, -1, self.C)  # (B*T, F, C)
        
        # compute cosine similarity
        x_folded_unsq = x_folded.unsqueeze(-1)  # (B*T, F, C, 1)
        sim_matrix = F.cosine_similarity(x_folded_unsq, x_folded_unsq.transpose(-2, -1), dim=1)  # (B*T, C, C)
        sim_matrix = sim_matrix.unsqueeze(1)  # (B*T, 1, C, C)

        # apply kernels
        m = self.kernels(sim_matrix)  # (B*T, A)
        p = self.position_kernels(sim_matrix)  # (B*T, P*A)
        m = m.view(B, T, self.A)  # (B, T, A)
        p = p.view(B, T, self.P, self.A)  # (B, T, P, A)

        # compute softmax over algebra dimension
        m = F.softmax(m, dim=-1)  # (B, T, A)

        # compute weighted sum over algebra dimension to get position output
        p = p.view(B, T, self.P, self.A)  # (B, T, P, A)
        out = torch.einsum('btpa, bta -> btp', p, m)  # (B, T, P)

        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    context_size = 11
    history_steps = 21

    model = Algebra_Core(position_output_dim=2, num_algebras=12, context_size=context_size).to(device)
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