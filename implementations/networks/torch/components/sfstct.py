import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

try:
    from ..components.base import init_weights
except ImportError:
    from implementations.networks.torch.components.base import init_weights


class SpatialEncoder(nn.Module):

    def __init__(self, img_size=64, patch_size=4, in_chans=4, embed_dim=512, depth=6, num_heads=8, mlp_ratio=4., dropout=0.1, vector_dim=128):
        super().__init__()
        
        # 1. Patch Embedding (Standard ViT logic) 
        # 64x64 image / 4x4 patch = 16x16 grid = 256 tokens
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 2. Vector Projection (The "State" Token) 
        # Projects 128-bit vector to 512-dim token
        self.vector_proj = nn.Linear(vector_dim, embed_dim)
        
        # 3. Spatial Positional Embedding
        # +1 for the vector token. Shape: (1, 257, 512)
        self.pos_drop = nn.Dropout(p=dropout)
        
        # 4. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=int(embed_dim * mlp_ratio), 
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self.reset_parameters()


    def reset_parameters(self):
        """
        Initializes weights using Truncated Normal distribution (std=0.02)
        following ViT best practices.
        """
        # Apply init to all sub-modules (Linear, Conv2d, LayerNorm)
        self.apply(init_weights)


    def forward(self, x, vec):
        # x: (B, T, C, H, W) -> flattened to (B*T, C, H, W) for spatial processing
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w) 
        vec = vec.view(b * t, -1)

        # Create Patch Tokens
        x = self.patch_embed(x) # (B*T, 512, 16, 16)
        x = x.flatten(2).transpose(1, 2) # (B*T, 256, 512)

        # Create Vector Token
        v_token = self.vector_proj(vec).unsqueeze(1) # (B*T, 1, 512)

        # Concatenate:
        x = torch.cat((v_token, x), dim=1) # (B*T, 257, 512)

        # Add Positional Embedding & Dropout
        x = self.pos_drop(x)

        # Apply Transformer
        x = self.blocks(x)
        x = self.norm(x)

        # Extract only the 0-th token (the contextualized vector token)
        # This token now contains info from the vector AND the image patches
        cls_token = x[:, 0] # (B*T, 512)

        # Reshape back to temporal sequence: (B, T, 512)
        return cls_token.view(b, t, -1)


class TemporalEncoder(nn.Module):

    def __init__(self, embed_dim=512, depth=4, num_heads=8, mlp_ratio=4., dropout=0.1):
        super().__init__()
        
        # Temporal Positional Embedding (encodes t-16... t)
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=int(embed_dim * mlp_ratio), 
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.reset_parameters()


    def reset_parameters(self):
        self.apply(init_weights)


    def forward(self, x):
        # x: (B, T, 512)
        B, T, D = x.shape
        
        # Add Temporal Position
        x = self.pos_drop(x)

        # Generate Causal Mask [7]
        # Ensures frame t cannot attend to frame t+1
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        
        # Apply Transformer with mask
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.norm(x)
        
        # Return only the last token (frame t) for prediction
        return x 


class SF_STCT(nn.Module):

    def __init__(self):
        super().__init__()
        
        # Configuration matches the report specification
        self.spatial_encoder = SpatialEncoder(
            img_size=64, patch_size=4, vector_dim=128, embed_dim=512, depth=6
        )
        
        self.temporal_encoder = TemporalEncoder(
            embed_dim=512, depth=4
        )

        # Prediction Heads [10]
        # Decoupling MLP before projection is best practice
        self.head_x = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 64) # 64 classes
        )
        self.head_y = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 64) # 64 classes
        )
        self.head_action = nn.Sequential(
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 6)   # 6 classes
        )
        self.head_flag = nn.Sequential(
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 2)   # 2 classes
        )
        self.head_value = nn.Sequential(
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Linear(64, 1)   # Regression output
        )


    def reset_parameters(self):
        """
        Public method to reset the entire model.
        """
        self.spatial_encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.head_x.apply(init_weights)
        self.head_y.apply(init_weights)
        self.head_action.apply(init_weights)
        self.head_flag.apply(init_weights)
        self.head_value.apply(init_weights)


    def forward(self, img, vec):
        """
        img: (Batch, Frames, 3, 64, 64) - normalized floats
        vec: (Batch, Frames, 128) - floats
        """
        # 1. Spatial Processing (Independent per frame)
        spatial_feats = self.spatial_encoder(img, vec) # Output: (B, T, 512)
        
        # 2. Temporal Processing (Across frames with causal mask)
        temporal_feat = self.temporal_encoder(spatial_feats) # Output: (B, T, 512)
        
        # 3. Task Specific Heads (Classification)
        logits_x = self.head_x(temporal_feat)      # (B, T, 64)
        logits_y = self.head_y(temporal_feat)      # (B, T, 64)
        logits_action = self.head_action(temporal_feat) # (B, T, 6)
        logits_flag = self.head_flag(temporal_feat)    # (B, T, 2)
        logits_value = self.head_value(temporal_feat)    # (B, T, 1)
    
        return logits_x, logits_y, logits_action, logits_flag, logits_value
    

# --- Example Usage ---
if __name__ == "__main__":
    # Batch of 4, 16 history + 1 current = 17 frames
    # Images should be floats (e.g., divided by 255.0)
    dummy_img = torch.randn(4, 17, 4, 64, 64)
    dummy_vec = torch.randn(4, 17, 128)

    model = SF_STCT()
    model.reset_parameters()
    x_out, y_out, action_out, flag_out, value_out = model(dummy_img, dummy_vec)

    print(f"X Logits: {x_out.shape}")
    print(f"Y Logits: {y_out.shape}")
    print(f"Action Logits: {action_out.shape}")
    print(f"Flag Logits: {flag_out.shape}")
    print(f"Value Output: {value_out.shape}")