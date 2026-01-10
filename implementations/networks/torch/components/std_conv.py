import torch
import torch.nn as nn

try:
    from ..components.base import init_weights
except ImportError:
    from implementations.networks.torch.components.base import init_weights


class ImpalaBlock(nn.Module):
    """
    A single block of the IMPALA architecture.
    Structure: Conv -> MaxPool -> ResBlock -> ResBlock
    """
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        
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
        super(ImpalaCNN, self).__init__()
        
        self.input_channels = input_channels
        self.output_dims = output_dims
        self.width = width
        self.height = height

        self.layers = nn.ModuleList()
        
        # Build the 3 main blocks
        current_channels = input_channels
        for depth in depths:
            self.layers.append(ImpalaBlock(current_channels, depth))
            current_channels = depth
            
        self.relu = nn.ReLU()
        
        # Calculate Flatten Dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, height, width)
            for layer in self.layers:
                dummy = layer(dummy)
            dummy = self.relu(dummy)
            self.flatten_dim = dummy.view(1, -1).size(1)
            
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
        
        x = self.relu(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x
    

if __name__ == "__main__":
    model = ImpalaCNN(output_dims=256, input_channels=3, width=84, height=84)
    model.reset_parameters()
    x = torch.randn(2, 3, 84, 84)
    out = model(x)
    assert out.shape == (2, 256)