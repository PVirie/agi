import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .base import init_weights
except ImportError:
    from implementations.core.torch.base import init_weights


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_groups=min(32, mid_channels), num_channels=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
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
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2], mode='reflect')
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DeepTemporalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        # We use TransformerEncoder with a Causal Mask to implement a 
        # "Decoder-only" (GPT-style) temporal model.
        # This allows frames to attend to history through multiple layers.
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            # Since input_dim is large (~4000), we keep feedforward dim equal to input_dim
            # to prevent parameter explosion, but standard is usually 4 * input_dim.
            dim_feedforward=input_dim, 
            dropout=dropout,
            batch_first=True,
            norm_first=True # Improves gradient flow in deep networks
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x: [Batch, Time, Features]
        """
        B, T, C = x.shape
        
        # --- Causal Mask ---
        # Returns [T, T] float matrix. 
        # 0.0 for visible (past/present), -inf for masked (future).
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        
        # Pass through the deep transformer stack
        out = self.transformer(x, mask=mask)
        return out



class TemporalUNet(nn.Module):
    def __init__(self, n_channels=1, vec_dim=128, num_temporal_layers=2, bilinear=True):
        super(TemporalUNet, self).__init__()
        self.n_channels = n_channels
        self.vec_dim = vec_dim
        self.bilinear = bilinear

        # --- Encoder ---
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)

        # --- Temporal Bottleneck ---
        self.bottleneck_channels = 512 // factor
        self.flat_features = self.bottleneck_channels * 4 * 4
        self.attn_input_dim = self.flat_features + vec_dim
        
        self.temporal_attn = DeepTemporalTransformer(
            input_dim=self.attn_input_dim,
            num_heads=self.attn_input_dim,
            num_layers=num_temporal_layers,
            dropout=0.1
        )
        self.fusion = nn.Linear(self.attn_input_dim, self.flat_features)

        # --- Decoder ---
        self.up1 = Up(self.bottleneck_channels, 256 // factor, 256, bilinear)
        self.up2 = Up(256 // factor, 128 // factor, 128, bilinear)
        self.up3 = Up(128 // factor, 64 // factor, 64, bilinear)
        self.up4 = Up(64 // factor, 32, 32, bilinear)

        self.head_heatmap = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )
        self.head_content = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, n_channels, kernel_size=1)
        )
        
        self.reset_parameters()


    def reset_parameters(self):
        self.apply(init_weights)


    def _get_max_features(self, feature_map, heatmap_logits):
        """
        Extracts features from the feature_map at the location of the 
        max value in heatmap_logits.
        """
        N, C, H, W = feature_map.shape
        
        # Flatten spatial dims
        heatmap_flat = heatmap_logits.view(N, -1)
        
        # Soft Argmax to find expected feature vector
        heatmap_probs = F.softmax(heatmap_flat, dim=1) # [N, H*W]
        features_flat = feature_map.view(N, C, -1)
        selected_features = torch.bmm(features_flat, heatmap_probs.unsqueeze(2)).squeeze(2) # [N, C]

        return selected_features


    def forward(self, x, v):
        """
        x: [Batch, Time, Channels, Height, Width]
        v: [Batch, Time, VecDim]
        Returns:
            x_log_probs: [Batch, Time, 64] (Log Probabilities for X coordinate)
            y_log_probs: [Batch, Time, 64] (Log Probabilities for Y coordinate)
            flag_logits: [Batch, Time, 6]  (Logits for Flag)
        """
        B, T, C, H, W = x.shape
        
        # --- U-Net Encoder/Decoder (Batch+Time folded) ---
        x_reshaped = x.view(B * T, C, H, W)
        x1 = self.inc(x_reshaped)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Temporal Attention
        flat_x5 = x5.view(B * T, -1)
        v_reshaped = v.view(B * T, -1)
        combined = torch.cat([flat_x5, v_reshaped], dim=1)
        combined = combined.view(B, T, -1)
        attn_out = self.temporal_attn(combined)
        
        fused = self.fusion(attn_out.view(B * T, -1)) # [B*T, flat_features]
        x_dec = fused.view_as(x5)
        
        x = self.up1(x_dec, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x_features = self.up4(x, x1) # [B*T, 32, 64, 64]
        
        # --- Generate Heatmap ---
        heatmap_logits = self.head_heatmap(x_features) # [B*T, 1, 64, 64]
        
        # --- Compute direct content ---
        content_logits = self.head_content(x_features) # [B*T, C, 64, 64]
        
        # --- Reshape and Return ---
        sampled_features = fused.view(B, T, -1) # [B, T, flat_features]
        heatmap_logits = heatmap_logits.view(B, T, H, W)
        content_logits = content_logits.view(B, T, C, H, W)
        
        return sampled_features, heatmap_logits, content_logits


if __name__ == "__main__":
    # Test
    model = TemporalUNet(n_channels=1, vec_dim=128, num_temporal_layers=2, bilinear=True)
    img = torch.randn(2, 5, 1, 64, 64)
    vec = torch.randn(2, 5, 128)
    
    features, heatmap_logits, content_logits = model(img, vec)
    
    assert features.shape == (2, 5, model.flat_features)
    assert heatmap_logits.shape == (2, 5, 64, 64)
    assert content_logits.shape == (2, 5, 1, 64, 64)

    print("Forward pass successful.")