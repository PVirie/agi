import torch
import torch.nn as nn
import torch.nn.functional as F

from implementations.networks.torch.components.wavenet import Wavenet
from implementations.networks.torch.components.std_conv import ImpalaBlock
from implementations.networks.torch.components.base import init_weights


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.post_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x, skip_connection):
        x = self.up_sample(x)
        x = self.pre_conv(x)
        x = x + skip_connection
        x = self.post_conv(x)
        return x
    

class TemporalUNet(nn.Module):
    def __init__(self, 
            output_dims, input_channels=1, 
            width=64, height=64, vec_dim=128, 
            hidden_dim=32, depths=[16, 32, 32], 
            history_steps=1, max_temporal_len=32):
        super().__init__()

        self.output_dims = output_dims
        self.n_channels = input_channels
        self.width = width
        self.height = height
        self.vec_dim = vec_dim
        
        self.layers = nn.ModuleList()
        
        # Build the 3 main blocks
        current_channels = input_channels
        for depth in depths:
            self.layers.append(ImpalaBlock(current_channels, depth))
            current_channels = depth
            
        # Calculate shape dynamically
        self.shapes = []
        with torch.no_grad():
            dummy = torch.zeros(input_channels, height, width)
            self.shapes.append(dummy.shape)
            for layer in self.layers:
                dummy = layer(dummy)
                self.shapes.append(dummy.shape)
            self.last_flatten_dim = dummy.reshape(1, -1).size(1)

        # projectors
        self.forward_proj = nn.Sequential(
            nn.Linear(self.last_flatten_dim + vec_dim, hidden_dim)
        )

        self.temporal_attn = Wavenet(
            d_model=hidden_dim,
            num_layers=1, # can only use 1 layer to not violate history constraint
            history_steps=history_steps
        )

        self.backward_proj = nn.Sequential(
            nn.Linear(hidden_dim, self.last_flatten_dim)
        )

        # --- Decoder ---
        self.up_c = nn.ModuleList()
        self.up_h = nn.ModuleList()

        for i in range(len(self.shapes)-1, 0, -1):
            in_ch, out_ch = self.shapes[i][0], self.shapes[i-1][0]
            self.up_c.append(UpConvBlock(in_ch, out_ch))
            self.up_h.append(UpConvBlock(in_ch, out_ch))

        self.head_feature = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dims),
        )
        self.head_heatmap = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(input_channels, 1, kernel_size=3, padding=1)
        )
        self.head_content = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        )
        
        self.reset_parameters()


    def reset_parameters(self):
        self.apply(init_weights)


    def forward(self, x, v):
        """
        x: [Batch, Time, Channels, Height, Width]
        v: [Batch, Time, VecDim]
        Returns:
            sampled_features: [Batch, Time, output_dims]
            x_logits: [Batch, Time, Width]
            y_logits: [Batch, Time, Height]
            content_logits: [Batch, Time, Channels, Height, Width]
        """
        B, T, C, H, W = x.shape
        
        # --- U-Net Encoder (Batch+Time folded) ---
        x_reshaped = x.reshape(B * T, C, H, W)

        skip_connections = []
        out = x_reshaped
        for layer in self.layers:
            skip_connections.append(out)
            out = layer(out)
        x_last = out  # [B*T, last_channels, last_height, last_width]

        # 2. Flatten spatial dims
        flat_last = x_last.reshape(B, T, -1) # [B, T, last_flatten_dim]
        
        combined = torch.cat([flat_last, v], dim=-1)
        projected = self.forward_proj(combined) # [B, T, hidden_dim]

        # # 3. Pass through Transformer
        attn_out = self.temporal_attn(projected) # [B, T, hidden_dim]
        
        sampled_features = self.head_feature(attn_out) # [B*T, output_dims]

        # 4. Fuse and Reshape
        flat_backward = self.backward_proj(attn_out.reshape(B * T, -1)) # [B*T, last_flatten_dim]

        backward = flat_backward.reshape(B * T, self.shapes[-1][0], self.shapes[-1][1], self.shapes[-1][2]) # [B*T, C, H, W]
        
        xh_features = backward
        xc_features = backward

        # --- U-Net Decoder ---
        for up_c_block, up_h_block, skip in zip(self.up_c, self.up_h, reversed(skip_connections)):
            xc_features = up_c_block(xc_features, skip)
            xh_features = up_h_block(xh_features, skip)

        # --- Generate Heatmap ---
        heatmap_logits = self.head_heatmap(xh_features) # [B*T, 1, H, W]
        
        # max over Height (dim 2) -> X distribution (Width)
        x_logits = heatmap_logits.max(dim=2).values # [B*T, 1, W]

        # max over Width (dim 3) -> Y distribution (Height)
        y_logits = heatmap_logits.max(dim=3).values # [B*T, 1, H]
        
        # --- Compute direct content ---
        content_logits = self.head_content(xc_features) # [B*T, C, H, W]
        
        # --- Reshape and Return ---
        sampled_features = sampled_features.reshape(B, T, self.output_dims)
        x_logits = x_logits.reshape(B, T, W)
        y_logits = y_logits.reshape(B, T, H)
        content_logits = content_logits.reshape(B, T, C, H, W)
        
        return sampled_features, x_logits, y_logits, content_logits


if __name__ == "__main__":
    # max_temporal_len defaults to 32, we pass 32 to be explicit or test with it.
    model = TemporalUNet(output_dims=16, input_channels=3, width=32, height=64, vec_dim=128, hidden_dim=16, depths=[16, 32, 32], history_steps=2, max_temporal_len=32)
    img = torch.randn(2, 5, 3, 64, 32)
    vec = torch.randn(2, 5, 128)
    
    features, x_logits, y_logits, content_logits = model(img, vec)
    
    assert features.shape == (2, 5, model.output_dims)
    assert x_logits.shape == (2, 5, 32)
    assert y_logits.shape == (2, 5, 64)
    assert content_logits.shape == (2, 5, 3, 64, 32)

    print("Forward pass successful.")

    # now test optimizer step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = features.sum() + x_logits.sum() + y_logits.sum() + content_logits.sum()
    loss.backward()
    optimizer.step()
    print("Optimizer step successful.")