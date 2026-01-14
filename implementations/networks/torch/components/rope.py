import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Rotary Embeddings (Optimized)
# ==========================================
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000, device=None):
        super().__init__()
        # Force float32 for frequency calculation to avoid numerical instability
        self.inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, device=device).float() / d))
        self.register_buffer("inv_freq_buffer", self.inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        """
        x: (Batch, Seq_Len, Heads, Head_Dim) or (Batch, Seq_Len, Dim)
        Returns: cos, sin with shape (1, Seq_Len, 1, Head_Dim)
        """
        seq_len = x.shape[seq_dim]
        
        # Refresh cache if sequence length changes
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            # Use the device of the input tensor
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq_buffer)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq_buffer)
            # Concat to match dimension of head_dim (d/2 + d/2)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Reshape for broadcasting: (1, Seq, 1, Dim)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]

        return self.cos_cached, self.sin_cached

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    # Ensure cos/sin are cast to x's dtype (e.g., if x is fp16)
    return (x * cos.to(x.dtype)) + (rotate_half(x) * sin.to(x.dtype))


# ==========================================
# 2. Attention Block (RoPE-Aware)
# ==========================================
class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, is_cross_attention=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.is_cross_attention = is_cross_attention
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Only init RoPE for Self-Attention
        if not self.is_cross_attention:
            self.rope = RotaryPositionalEmbeddings(self.head_dim)
        
        self.dropout = dropout

    def forward(self, x, context=None, mask=None):
        """
        mask: Tensor with 0.0 for keep and -inf for mask.
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Projections
        # Shape: (Batch, Seq, Heads, Head_Dim)
        q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim)
        
        kv_input = context if context is not None else x
        k = self.k_proj(kv_input).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(kv_input).view(batch_size, -1, self.num_heads, self.head_dim)

        # 2. Apply RoPE (Self-Attention Only)
        if not self.is_cross_attention:
            cos, sin = self.rope(q, seq_dim=1)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # 3. Transpose for Attention: (Batch, Heads, Seq, Head_Dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. Scaled Dot Product Attention
        # is_causal is ALWAYS False here because we handle causality via the 'mask' argument
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask, 
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False 
        )

        # 5. Output Projection
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


# ==========================================
# 3. Layers (Pre-Norm for RL Stability)
# ==========================================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = RoPEMultiHeadAttention(d_model, num_heads, dropout, is_cross_attention=False)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout(self.attn(x2, mask=mask))
        x2 = self.norm2(x)
        x = x + self.ffn(x2)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(d_model, num_heads, dropout, is_cross_attention=False)
        self.cross_attn = RoPEMultiHeadAttention(d_model, num_heads, dropout, is_cross_attention=True)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, tgt_mask=None, src_mask=None):
        # Self Attention (tgt_mask contains Causal + History logic)
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, mask=tgt_mask))
        
        # Cross Attention
        x2 = self.norm2(x)
        x = x + self.dropout(self.cross_attn(x2, context=context, mask=src_mask))
        
        # Feed Forward
        x2 = self.norm3(x)
        x = x + self.ffn(x2)
        return x


# ==========================================
# 4. Main Transformer Model
# ==========================================
class RoPETransformer(nn.Module):
    def __init__(self, d_model=256, num_heads=4, num_layers=2, d_ff=1024, dropout=0.1, history_steps=None):
        super().__init__()
        self.history_steps = history_steps
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)

    def _generate_mask(self, size, device):
        """Generates the causal + history limited mask."""
        # Start with -inf (Masked)
        mask = torch.full((size, size), float('-inf'), device=device)
        
        if self.history_steps == 0:
            # Diagonal only
            mask.fill_diagonal_(0.0)
        else:
            # 1. Causality: Lower triangle (keep)
            valid = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))

            # 2. History Limit: Upper triangle relative to diagonal -N
            if self.history_steps is not None:
                # We want to keep cols >= row - history_steps
                history_constraint = torch.triu(
                    torch.ones(size, size, dtype=torch.bool, device=device), 
                    diagonal=-self.history_steps
                )
                valid = valid & history_constraint
                
            # Set valid positions to 0.0 (Keep)
            mask.masked_fill_(valid, 0.0)

        return mask

    def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None):
        """
        src: (Batch, Src_Len, Dim)
        tgt: (Batch, Tgt_Len, Dim)
        src_pad_mask: (Batch, 1, 1, Src_Len) - Float mask (0.0 keep, -inf block)
        tgt_pad_mask: (Batch, 1, 1, Tgt_Len) - Float mask (0.0 keep, -inf block)
        """
        device = src.device
        
        # --- Decoder Mask Construction ---
        T = tgt.shape[1]
        
        # 1. Structural Mask (Causal + History) - Shape (T, T)
        structure_mask = self._generate_mask(T, device)
        
        # 2. Combine with Padding Mask
        if tgt_pad_mask is not None:
            # Broadcast: (T, T) + (B, 1, 1, T) -> (B, 1, T, T)
            # -inf + 0 = -inf (Masked)
            # 0 + 0 = 0 (Kept)
            final_tgt_mask = structure_mask + tgt_pad_mask
        else:
            final_tgt_mask = structure_mask

        # --- Forward Passes ---
        
        # Encoder
        enc_out = src
        for layer in self.encoder_layers:
            # Encoder usually only needs padding mask
            enc_out = layer(enc_out, mask=src_pad_mask)
            
        # Decoder
        dec_out = tgt
        for layer in self.decoder_layers:
            dec_out = layer(
                dec_out, 
                context=enc_out, 
                tgt_mask=final_tgt_mask, 
                src_mask=src_pad_mask # Source mask applies to Cross Attention keys
            )
            
        return self.norm(dec_out)

# ==========================================
# 5. Helper Function for Users
# ==========================================
def create_padding_mask(tensor, pad_idx=0):
    """
    Creates a float mask for SDPA.
    Returns: (Batch, 1, 1, Seq_Len)
    """
    batch_size, seq_len = tensor.shape
    # True where value is NOT padding
    valid = (tensor != pad_idx).unsqueeze(1).unsqueeze(2) 
    mask = torch.zeros((batch_size, 1, 1, seq_len), device=tensor.device)
    mask = mask.masked_fill(~valid, float("-inf"))
    return mask


# --- Verification ---
if __name__ == "__main__":
    # Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoPETransformer(d_model=64, num_heads=4, num_layers=2, history_steps=3).to(device)
    
    # Fake Data (Batch=2, Seq=8)
    src_tokens = torch.randint(1, 100, (2, 8)).to(device)
    tgt_tokens = torch.randint(1, 100, (2, 8)).to(device)
    
    # Add padding to 2nd sample
    src_tokens[1, 5:] = 0
    tgt_tokens[1, 5:] = 0
    
    # Embeddings
    emb = nn.Embedding(100, 64).to(device)
    src_emb = emb(src_tokens)
    tgt_emb = emb(tgt_tokens)
    
    # Create Padding Masks
    src_mask = create_padding_mask(src_tokens, pad_idx=0)
    tgt_mask = create_padding_mask(tgt_tokens, pad_idx=0)
    
    assert src_emb.shape == (2, 8, 64)
    assert tgt_emb.shape == (2, 8, 64)

    # Forward
    out = model(src_emb, tgt_emb, src_pad_mask=src_mask, tgt_pad_mask=tgt_mask)
    assert out.shape == (2, 8, 64)

    print(f"Output shape: {out.shape}")
    print("Mask logic verified: -inf used for masking, 0.0 for keeping.")
