import torch
import torch.nn as nn
import torch.nn.functional as F

from implementations.networks.torch.components.rope import RoPEDecoderOnly
from implementations.networks.torch.components.base import init_weights


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, skip_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle arbitrary sizes by padding x1 to match x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2], mode='reflect')
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TemporalUNet(nn.Module):
    def __init__(self, n_channels=1, vec_dim=128, num_temporal_layers=2, hidden_dim=32, bilinear=True, history_steps=8, max_temporal_len=32):
        super(TemporalUNet, self).__init__()
        self.n_channels = n_channels
        self.vec_dim = vec_dim
        self.bilinear = bilinear
        
        # --- Config for Arbitrary Input Support ---
        # We force the bottleneck to be 4x4 so the Linear layers work 
        # regardless of the actual input image aspect ratio or size.
        self.bottleneck_size = 2

        f1 = n_channels * 4
        f2 = n_channels * 8
        f3 = n_channels * 16
        f4 = n_channels * 32
        f5 = n_channels * 64

        # --- Encoder ---
        self.inc = DoubleConv(n_channels, f1)
        self.down1 = Down(f1, f2)
        self.down2 = Down(f2, f3)
        self.down3 = Down(f3, f4)
        factor = 2 if bilinear else 1
        self.down4 = Down(f4, f5 // factor)

        # Flat feature projector
        self.temporal_proj = nn.Linear(vec_dim, hidden_dim)

        # --- Temporal Bottleneck ---
        self.bottleneck_channels = f5 // factor
        
        # Fixed size feature map for Transformer (C * 4 * 4)
        self.flat_features = self.bottleneck_channels * self.bottleneck_size * self.bottleneck_size
        self.attn_input_dim = self.flat_features + hidden_dim
        
        self.temporal_attn = RoPEDecoderOnly(
            d_model=self.attn_input_dim, 
            num_heads=8, 
            num_layers=num_temporal_layers, 
            d_ff=1024, 
            dropout=0.1, 
            history_steps=history_steps
        )

        self.fusion = nn.Linear(self.attn_input_dim, self.flat_features)

        # --- Decoder ---
        self.up_c_1 = Up(self.bottleneck_channels, f4 // factor, f4, bilinear)
        self.up_c_2 = Up(f4 // factor, f3 // factor, f3, bilinear)
        self.up_c_3 = Up(f3 // factor, f2 // factor, f2, bilinear)
        self.up_c_4 = Up(f2 // factor, f1, f1, bilinear)

        # --- Decoder ---
        self.up_h_1 = Up(self.bottleneck_channels, f4 // factor, f4, bilinear)
        self.up_h_2 = Up(f4 // factor, f3 // factor, f3, bilinear)
        self.up_h_3 = Up(f3 // factor, f2 // factor, f2, bilinear)
        self.up_h_4 = Up(f2 // factor, f1, f1, bilinear)

        self.out_features = hidden_dim
        self.head_feature = nn.Sequential(
            nn.Linear(self.flat_features, self.out_features),
            nn.LayerNorm(self.out_features),
            nn.ReLU()
        )
        self.head_heatmap = nn.Sequential(
            nn.Conv2d(f1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )
        self.head_content = nn.Sequential(
            nn.Conv2d(f1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, n_channels, kernel_size=1)
        )
        
        self.reset_parameters()


    def reset_parameters(self):
        self.apply(init_weights)


    def forward(self, x, v):
        """
        x: [Batch, Time, Channels, Height, Width]
        v: [Batch, Time, VecDim]
        Returns:
            sampled_features: [Batch, Time, out_features]
            x_logits: [Batch, Time, Width]
            y_logits: [Batch, Time, Height]
            content_logits: [Batch, Time, Channels, Height, Width]
        """
        B, T, C, H, W = x.shape
        
        # --- U-Net Encoder (Batch+Time folded) ---
        x_reshaped = x.reshape(B * T, C, H, W)
        x1 = self.inc(x_reshaped)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # Shape: [B*T, features, H/16, W/16]

        # --- Temporal Attention Input Construction ---
        # 1. Adaptive Pool to fixed size (4x4) to handle arbitrary input H, W
        # # REVISION: Replaced adaptive_avg_pool2d with F.interpolate for determinism.
        # We use 'bilinear' to resize the feature map to the fixed 4x4 bottleneck size.
        # This acts as a spatial compression step compatible with any input resolution.
        x5_pooled = F.interpolate(
            x5, 
            size=(self.bottleneck_size, self.bottleneck_size), 
            mode='bilinear', 
            align_corners=True
        )

        # 2. Flatten spatial dims
        flat_x5 = x5_pooled.reshape(B, T, -1) # [B, T, flat_features]
        
        projected_v = self.temporal_proj(v) # [B, T, hidden_dim]
        combined = torch.cat([flat_x5, projected_v], dim=2)

        # 3. Pass through Transformer
        attn_out = self.temporal_attn(combined)
        
        # 4. Fuse and Reshape
        fused = self.fusion(attn_out.reshape(B * T, -1)) # [B*T, flat_features]
        x4_small = fused.reshape(B * T, self.bottleneck_channels, self.bottleneck_size, self.bottleneck_size)
        
        # 5. Interpolate back to original bottleneck spatial size (H/16, W/16)
        # This is crucial: x5 spatial size might be e.g., (2, 4) if input is 32x64
        x4_ = F.interpolate(x4_small, size=x5.shape[-2:], mode='bilinear', align_corners=True)
        
        # --- U-Net Decoder ---
        xc3_ = self.up_c_1(x4_, x4)
        xc2_ = self.up_c_2(xc3_, x3)
        xc1_ = self.up_c_3(xc2_, x2)
        xc_features = self.up_c_4(xc1_, x1) # [B*T, C, H, W]
        
        sampled_features = self.head_feature(fused) # [B*T, out_features]

        xh3_ = self.up_h_1(x4_, x4)
        xh2_ = self.up_h_2(xh3_, x3)
        xh1_ = self.up_h_3(xh2_, x2)
        xh_features = self.up_h_4(xh1_, x1) # [B*T, C, H, W]

        # --- Generate Heatmap ---
        heatmap_logits = self.head_heatmap(xh_features) # [B*T, 1, H, W]
        
        # max over Height (dim 2) -> X distribution (Width)
        x_logits = heatmap_logits.max(dim=2).values # [B*T, 1, W]

        # max over Width (dim 3) -> Y distribution (Height)
        y_logits = heatmap_logits.max(dim=3).values # [B*T, 1, H]
        
        # --- Compute direct content ---
        content_logits = self.head_content(xc_features) # [B*T, C, H, W]
        
        # --- Reshape and Return ---
        sampled_features = sampled_features.reshape(B, T, self.out_features)
        x_logits = x_logits.reshape(B, T, W)
        y_logits = y_logits.reshape(B, T, H)
        content_logits = content_logits.reshape(B, T, C, H, W)
        
        return sampled_features, x_logits, y_logits, content_logits


if __name__ == "__main__":
    # max_temporal_len defaults to 32, we pass 32 to be explicit or test with it.
    model = TemporalUNet(n_channels=3, vec_dim=128, num_temporal_layers=2, hidden_dim=16, bilinear=True, history_steps=2, max_temporal_len=32)
    img = torch.randn(2, 5, 3, 64, 32)
    vec = torch.randn(2, 5, 128)
    
    features, x_logits, y_logits, content_logits = model(img, vec)
    
    assert features.shape == (2, 5, model.out_features)
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