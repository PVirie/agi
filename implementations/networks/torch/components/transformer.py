import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from implementations.networks.torch.components.base import init_weights


class InstructionTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, max_len, mlp_ratio=2, dropout=0.01):
        super().__init__()
        self.d_model = d_model
        
        # 1. Positional Encoding: A learned parameter for each position up to max_len
        # Shape: (1, max_len, d_model) for easy broadcasting
        # self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))

        self.dropout = nn.Dropout(p=dropout)

        self.input_projection = nn.Linear(input_dim, d_model)  # Optional: project input to d_model if needed
        
        # 2. Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        

    def reset_parameters(self):
        self.apply(init_weights)

        # init positional encoding to small values to prevent early saturation
        # trunc_normal_(self.pos_embedding, std=0.02)


    def forward(self, x, mask):
        """
        Args:
            x: Input tensor of shape (Batch, Seq, L, E)
            mask: Binary mask (1 for real, 0 for padding) of shape (Batch, Seq, L)
        Returns:
            pooled_output: (Batch, Seq, E)
        """
        B, S, L, E = x.shape
        
        x_reshaped = x.view(B * S, L, E)
        x_reshaped = self.input_projection(x_reshaped)
        
        # remove positional encoding per latest knowledge
        # x_reshaped = x_reshaped + self.pos_embedding[:, :L, :]
        x_reshaped = self.dropout(x_reshaped)
        
        # Prepare Mask (Convert 1/0 to True/False for PyTorch)
        flat_mask = mask.view(B * S, L)
        src_key_padding_mask = (flat_mask == 0) # True means "mask this out"

        # Pass through Transformer
        encoded = self.transformer_encoder(x_reshaped, src_key_padding_mask=src_key_padding_mask)

        # --- MASKED MEAN POOLING ---
        # Ensure we don't include padding in our feature average
        masked_encoded = encoded * flat_mask.unsqueeze(-1)
        sum_encoded = torch.sum(masked_encoded, dim=1)
        token_counts = torch.sum(flat_mask, dim=1, keepdim=True).clamp(min=1e-9)
        
        pooled = sum_encoded / token_counts # (B*S, d_model)

        return pooled.view(B, S, self.d_model)


def get_padding_mask(x, pad_token_id):
    # x: (Batch, Seq, L)
    return (x != pad_token_id).float()  # 1 for real tokens, 0 for padding


# --- Example Usage ---
if __name__ == "__main__":
    # Test InstructionTransformer
    instr_transformer = InstructionTransformer(input_dim=8, d_model=256, nhead=4, num_layers=2, max_len=20)
    dummy_instr_input = torch.randn(3, 10, 20, 8)  # (Batch=3, Seq=10, L=20, E=8)
    dummy_mask = torch.ones(3, 10, 20)  # All tokens are valid (no padding)
    instr_output = instr_transformer(dummy_instr_input, dummy_mask)
    assert instr_output.shape == (3, 10, 256), f"Expected output shape (3, 10, 256), got {instr_output.shape}"

    # Test get_padding_mask
    dummy_token_input = torch.tensor([[[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]])  # (Batch=1, Seq=2, L=5)
    pad_token_id = 0
    padding_mask = get_padding_mask(dummy_token_input, pad_token_id)
    expected_mask = torch.tensor([[[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]], dtype=torch.float)
    assert torch.equal(padding_mask, expected_mask), f"Expected mask {expected_mask}, got {padding_mask}"