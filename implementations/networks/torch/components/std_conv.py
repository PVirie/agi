import torch
import torch.nn as nn

from implementations.networks.torch.components.base import init_weights


class ImpalaBlock(nn.Module):
    """
    A single block of the IMPALA architecture.
    Structure: Conv -> MaxPool -> ResBlock -> ResBlock
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.res1 = self._build_res_pair(out_channels)
        self.res2 = self._build_res_pair(out_channels)


    def _build_res_pair(self, channels):
        return nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = x + self.res1(x)
        x = x + self.res2(x)
        return x


class ImpalaCNN(nn.Module):
    """
    The IMPALA ResNet architecture.
    Standard configuration for Atari: Channels [16, 32, 32]
    """
    def __init__(self, output_dims, input_channels, width, height, depths=[16, 32, 32]):
        super().__init__()
        
        self.output_dims = output_dims
        self.input_channels = input_channels
        self.width = width
        self.height = height

        self.layers = nn.ModuleList()
        
        # Build the 3 main blocks
        current_channels = input_channels
        for depth in depths:
            self.layers.append(ImpalaBlock(current_channels, depth))
            current_channels = depth
            
        self.activation = nn.ReLU()
        
        # Calculate Flatten Dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, height, width)
            for layer in self.layers:
                dummy = layer(dummy)
            dummy = self.activation(dummy)
            self.flatten_dim = dummy.reshape(1, -1).size(1)
            
        # Final fully connected layer to output_dims (standard in IMPALA paper)
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, output_dims),
            nn.ReLU()
        )

    
    def reset_parameters(self):
        self.apply(init_weights)


    def forward(self, x):
        # x shape: (B, C, H, W)
        for layer in self.layers:
            x = layer(x)
        
        x = self.activation(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class MiniGridCNN(nn.Module):
    """
    A compact convolutional encoder for small symbolic grids (e.g. MiniGrid 7x7x3).

    Each input channel (object id, color, state) is a categorical map and is
    embedded with its own embedding table. The per-channel embeddings are
    concatenated along the feature dimension and fed through a stack of small
    2x2 convolutions with a single max-pool (following the standard
    rl-starter-files / torch-ac MiniGrid model), then flattened and projected
    to `output_dims`.
    """
    def __init__(self, output_dims, input_channels, width, height, vocab_size=256, embedding_dim=8):
        super().__init__()

        self.output_dims = output_dims
        self.input_channels = input_channels
        self.width = width
        self.height = height
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Per-channel embedding tables (each channel has its own categorical vocab)
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim) for _ in range(input_channels)
        ])
        feature_channels = input_channels * embedding_dim

        self.conv = nn.Sequential(
            nn.Conv2d(feature_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
        )

        # Compute flatten dim dynamically (2x2 convs + maxpool shrink the map)
        with torch.no_grad():
            dummy = torch.zeros(1, feature_channels, height, width)
            dummy = self.conv(dummy)
            self.flatten_dim = dummy.reshape(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, output_dims),
            nn.ReLU()
        )


    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

        # Initialization from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
        def init_params(m):
            classname = m.__class__.__name__
            if classname.find("Linear") != -1:
                m.weight.data.normal_(0, 1)
                m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(init_params)


    def forward(self, x):
        # x: tensor of shape (B, H, W, C) with per-channel categorical indices
        embedded = [self.embeddings[c](x[..., c].long()) for c in range(self.input_channels)]
        x = torch.cat(embedded, dim=-1)  # (B, H, W, C * embedding_dim)
        x = x.permute(0, 3, 1, 2)  # (B, C * embedding_dim, H, W)
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class ImpalaBlock1D(nn.Module):
    """
    A 1D version of the IMPALA block for processing sequences.
    Structure: Conv1D -> MaxPool1D -> ResBlock1D -> ResBlock1D
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.res1 = self._build_res_pair(out_channels)
        self.res2 = self._build_res_pair(out_channels)


    def _build_res_pair(self, channels):
        return nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1)
        )
    

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = x + self.res1(x)
        x = x + self.res2(x)
        return x
    

class ImpalaCNN1D(nn.Module):
    """
    A 1D version of the IMPALA CNN for processing sequences.
    """
    def __init__(self, output_dims, input_channels, seq_length, depths=[16, 32, 32]):
        super().__init__()
        
        self.output_dims = output_dims
        self.input_channels = input_channels
        self.seq_length = seq_length
        
        self.layers = nn.ModuleList()
        
        # Build the 3 main blocks
        current_channels = input_channels
        for depth in depths:
            self.layers.append(ImpalaBlock1D(current_channels, depth))
            current_channels = depth
            
        self.activation = nn.ReLU()
        
        # Calculate Flatten Dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, seq_length)
            for layer in self.layers:
                dummy = layer(dummy)
            dummy = self.activation(dummy)
            self.flatten_dim = dummy.reshape(1, -1).size(1)
            
        # Final fully connected layer to output_dims
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, output_dims),
            nn.ReLU()
        )


    def reset_parameters(self):
        self.apply(init_weights)


    def forward(self, x):
        # x shape: (B, C, L)
        for layer in self.layers:
            x = layer(x)
        
        x = self.activation(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = ImpalaCNN(output_dims=256, input_channels=3, width=32, height=64)
    model.reset_parameters()
    x = torch.randn(2, 3, 64, 32)
    out = model(x)
    assert out.shape == (2, 256)

    # now test optimizer step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = out.sum()
    loss.backward()
    optimizer.step()

    print("Optimizer step successful.")

    model = ImpalaCNN1D(output_dims=256, input_channels=3, seq_length=10)
    model.reset_parameters()
    x = torch.randn(2, 3, 10)
    out = model(x)
    assert out.shape == (2, 256)

    model = MiniGridCNN(output_dims=128, input_channels=3, width=7, height=7)
    model.reset_parameters()
    x = torch.randint(0, 256, (4, 7, 7, 3))
    out = model(x)
    assert out.shape == (4, 128)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = out.sum()
    loss.backward()
    optimizer.step()

    print("MiniGridCNN step successful.")