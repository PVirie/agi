import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from implementations.networks.torch.components.base import init_weights


class InstructionTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, max_len, mlp_ratio=2, dropout=0.01, use_pe=False, norm_first=True, pooling="mean"):
        """
        pooling: "mean" averages the real (non-padding) tokens;
                 "cls" prepends a learned token and returns its encoding, so the summary is
                 attention-weighted and can depend on token order beyond what the PE leaves after averaging.
        """
        super().__init__()
        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling}")
        self.d_model = d_model
        self.use_pe = use_pe
        self.pooling = pooling

        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 1. Positional Encoding: A learned parameter for each position up to max_len
        # Shape: (1, max_len, d_model) for easy broadcasting
        if self.use_pe:
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))

        self.dropout = nn.Dropout(p=dropout)

        self.input_projection = nn.Linear(input_dim, d_model)  # Optional: project input to d_model if needed
        
        # 2. Standard Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first
        )
        # pre-LN leaves the residual stream unnormalised, so it needs a final norm
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if norm_first else None,
            enable_nested_tensor=not norm_first,
        )
        

    def reset_parameters(self):
        self.apply(init_weights)

        # init positional encoding to small values to prevent early saturation
        if self.use_pe:
            trunc_normal_(self.pos_embedding, std=0.02)
        if self.pooling == "cls":
            trunc_normal_(self.cls_token, std=0.02)


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
        
        # Add positional encoding if enabled
        if self.use_pe:
            x_reshaped = x_reshaped + self.pos_embedding[:, :L, :]

        # Prepare Mask (Convert 1/0 to True/False for PyTorch)
        flat_mask = mask.view(B * S, L)

        if self.pooling == "cls":
            # the CLS token is always attendable, so all-padding sequences need no guard
            x_reshaped = torch.cat([self.cls_token.expand(B * S, -1, -1), x_reshaped], dim=1)
            attn_mask = torch.cat([torch.ones_like(flat_mask[:, :1]), flat_mask], dim=1)
        else:
            # Guard against all-padding sequences: softmax over all-masked keys produces
            # NaN (0/0), which poisons LayerNorm gradients. Force at least one token to
            # be attended per sequence; the original flat_mask is still used for pooling
            # so these forced tokens never contribute to the output.
            has_any_real = flat_mask.any(dim=1, keepdim=True)  # (B*S, 1) bool
            first_token_unmasked = torch.zeros_like(flat_mask)
            first_token_unmasked[:, 0] = 1.0
            attn_mask = torch.where(has_any_real, flat_mask, first_token_unmasked)

        x_reshaped = self.dropout(x_reshaped)
        src_key_padding_mask = (attn_mask == 0)  # True means "mask this out"

        # Pass through Transformer
        encoded = self.transformer_encoder(x_reshaped, src_key_padding_mask=src_key_padding_mask)

        if self.pooling == "cls":
            return encoded[:, 0, :].reshape(B, S, self.d_model)

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

    # Test training
    instr_transformer.train()
    optimizer = torch.optim.Adam(instr_transformer.parameters(), lr=1e-3)
    optimizer.zero_grad()

    for i in range(5):  # Run a few training steps
        instr_output = instr_transformer(dummy_instr_input, dummy_mask)
        loss = instr_output.mean()  # Dummy loss for testing
        loss.backward()
        optimizer.step()

    # test invalid padding mask (all zeros)
    dummy_invalid_mask = torch.zeros(3, 10, 20)  # All tokens are padding
    instr_output_invalid = instr_transformer(dummy_instr_input, dummy_invalid_mask)
    assert torch.allclose(instr_output_invalid, torch.zeros_like(instr_output_invalid)), "Expected output to be zeros for all-padding input, got {instr_output_invalid}"

    # test training with invalid padding mask
    instr_transformer.train()
    optimizer.zero_grad()
    instr_output_invalid = instr_transformer(dummy_instr_input, dummy_invalid_mask)
    loss_invalid = instr_output_invalid.mean()  # Dummy loss for testing
    loss_invalid.backward()
    optimizer.step()

    # test CLS pooling, including all-padding sequences (CLS keeps attention well-defined)
    cls_transformer = InstructionTransformer(input_dim=8, d_model=64, nhead=4, num_layers=2, max_len=20, use_pe=True, pooling="cls", dropout=0.0)
    cls_transformer.reset_parameters()
    cls_transformer.train()
    cls_optimizer = torch.optim.Adam(cls_transformer.parameters(), lr=1e-3)
    mixed_mask = torch.ones(3, 10, 20)
    mixed_mask[0] = 0.0
    mixed_mask[1, :, 5:] = 0.0
    cls_output = cls_transformer(dummy_instr_input, mixed_mask)
    assert cls_output.shape == (3, 10, 64), f"Expected output shape (3, 10, 64), got {cls_output.shape}"
    assert torch.isfinite(cls_output).all(), "CLS pooling produced non-finite output"
    cls_optimizer.zero_grad()
    cls_output.pow(2).mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in cls_transformer.parameters()), "CLS pooling produced non-finite gradients"
    cls_optimizer.step()

    # padding content must not leak into the CLS summary
    cls_transformer.eval()
    perturbed_input = dummy_instr_input.clone()
    perturbed_input[1, :, 5:] = torch.randn_like(perturbed_input[1, :, 5:])
    with torch.no_grad():
        out_a = cls_transformer(dummy_instr_input, mixed_mask)
        out_b = cls_transformer(perturbed_input, mixed_mask)
    assert torch.allclose(out_a[1], out_b[1], atol=1e-5), "Padding tokens leaked into the CLS output"

    # word order must change the CLS summary
    swapped_input = dummy_instr_input.clone()
    swapped_input[2, :, [0, 1]] = swapped_input[2, :, [1, 0]]
    with torch.no_grad():
        out_c = cls_transformer(swapped_input, mixed_mask)
    assert not torch.allclose(out_a[2], out_c[2], atol=1e-5), "CLS output is insensitive to token order"

    print("All tests passed!")