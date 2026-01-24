import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Core Utilities (RoPE, Attention, FFN)
# ==========================================

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000, device=None):
        super().__init__()
        # Force float32 for precise frequency calculation
        self.inv_freq = 1.0 / (base ** (torch.arange(0, d, 2, device=device).float() / d))
        self.register_buffer("inv_freq_buffer", self.inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq_buffer)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq_buffer)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]
        return self.cos_cached, self.sin_cached

def apply_rope(x, cos, sin):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos.to(x.dtype)) + (rotated * sin.to(x.dtype))

class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, is_cross_attention=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.is_cross_attention = is_cross_attention
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE only for Self-Attention
        if not self.is_cross_attention:
            self.rope = RotaryPositionalEmbeddings(self.head_dim)
        
        self.dropout = dropout

    def forward(self, x, context=None, mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, -1, self.num_heads, self.head_dim)
        
        kv_input = context if context is not None else x
        k = self.k_proj(kv_input).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(kv_input).view(batch_size, -1, self.num_heads, self.head_dim)

        if not self.is_cross_attention:
            cos, sin = self.rope(q, seq_dim=1)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model))

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

# ==========================================
# 2. Unified Mask Generation Strategy (Mixin)
# ==========================================

class MaskGenerationMixin:
    """
    Handles the creation of structural masks (Causality + History constraints)
    and combines them with user-provided padding masks.
    """
    def __init__(self, history_steps=None):
        self.history_steps = history_steps

    def _generate_structural_mask(self, size, device):
        """Generates the base mask for Causal + History logic."""
        mask = torch.full((size, size), float('-inf'), device=device)
        
        if self.history_steps == 0:
            mask.fill_diagonal_(0.0)
        else:
            # 1. Causality (Lower Triangle)
            valid = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
            
            # 2. History Constraint (Upper Triangle relative to sliding window)
            if self.history_steps is not None:
                history_constraint = torch.triu(
                    torch.ones(size, size, dtype=torch.bool, device=device), 
                    diagonal=-self.history_steps
                )
                valid = valid & history_constraint
            
            mask.masked_fill_(valid, 0.0)
        return mask

    def _combine_masks(self, base_mask, user_mask):
        """Combines structural mask with user padding mask."""
        if user_mask is None:
            return base_mask
        # Broadcast: (T, T) + (B, 1, 1, T) -> (B, 1, T, T)
        return base_mask + user_mask


# ==========================================
# 3. Model Architectures
# ==========================================

# --- A. Encoder-Decoder ---
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attn = RoPEMultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(d_model, num_heads, dropout, is_cross_attention=False)
        self.cross_attn = RoPEMultiHeadAttention(d_model, num_heads, dropout, is_cross_attention=True)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, tgt_mask=None, src_mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), context=context, mask=src_mask))
        x = x + self.ffn(self.norm3(x))
        return x


class RoPEEncoderDecoder(nn.Module, MaskGenerationMixin):
    def __init__(self, d_model=256, num_heads=4, num_layers=2, d_ff=1024, dropout=0.1, history_steps=None):
        nn.Module.__init__(self)
        MaskGenerationMixin.__init__(self, history_steps)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None):
        device = src.device
        
        # Prepare Decoder Mask (Structure + Padding)
        structural_mask = self._generate_structural_mask(tgt.shape[1], device)
        final_tgt_mask = self._combine_masks(structural_mask, tgt_pad_mask)
        
        # Encoder
        enc_out = src
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, mask=src_pad_mask) # Encoder only uses padding mask
            
        # Decoder
        dec_out = tgt
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, context=enc_out, tgt_mask=final_tgt_mask, src_mask=src_pad_mask)
            
        return self.norm(dec_out)


# --- B. Decoder-Only (GPT) ---
class GPTDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attn = RoPEMultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.ffn(self.norm2(x))
        return x


class RoPEDecoderOnly(nn.Module, MaskGenerationMixin):
    def __init__(self, d_model=256, num_heads=4, num_layers=2, d_ff=1024, dropout=0.1, history_steps=None):
        nn.Module.__init__(self)
        MaskGenerationMixin.__init__(self, history_steps)
        
        self.layers = nn.ModuleList([GPTDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None):
        device = x.device
        
        # Prepare Mask (Structure + Padding)
        structural_mask = self._generate_structural_mask(x.shape[1], device)
        final_mask = self._combine_masks(structural_mask, pad_mask)
        
        out = x
        for layer in self.layers:
            out = layer(out, mask=final_mask)
            
        return self.norm(out)

# ==========================================
# 4. Usage Example
# ==========================================
if __name__ == "__main__":
    # Helper to simulate user creating a padding mask
    def create_pad_mask(tokens, pad_idx=0):
        # (Batch, 1, 1, Seq)
        mask = (tokens != pad_idx).unsqueeze(1).unsqueeze(2)
        return torch.zeros_like(mask, dtype=torch.float).masked_fill(~mask, float('-inf'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Decoder-Only Model
    model = RoPEDecoderOnly(history_steps=5).to(device)
    
    # 2. Fake Data
    x = torch.randint(0, 100, (2, 20)).to(device) # Batch 2, Seq 20
    x[1, 15:] = 0 # Add padding to 2nd sample
    
    # 3. Create Padding Mask
    pad_mask = create_pad_mask(x, pad_idx=0)
    
    # 4. Embed and Forward
    emb = nn.Embedding(100, 256).to(device)(x)
    assert emb.shape == (2, 20, 256)
    output = model(emb, pad_mask=pad_mask)
    assert output.shape == (2, 20, 256)

    print(f"Decoder-Only Output: {output.shape}") # Should be (2, 20, 256)

    # now test without mask
    output_no_mask = model(emb, pad_mask=None)
    assert output_no_mask.shape == (2, 20, 256)

    # now test Encoder-Decoder
    model_ed = RoPEEncoderDecoder(history_steps=3).to(device)
    src = torch.randint(0, 100, (2, 15)).to(device)
    src[1, 10:] = 0
    tgt = torch.randint(0, 100, (2, 12)).to(device)
    tgt[1, 8:] = 0
    src_pad_mask = create_pad_mask(src, pad_idx=0)
    tgt_pad_mask = create_pad_mask(tgt, pad_idx=0)
    src_emb = nn.Embedding(100, 256).to(device)(src)
    tgt_emb = nn.Embedding(100, 256).to(device)(tgt)
    output_ed = model_ed(src_emb, tgt_emb, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
    assert output_ed.shape == (2, 12, 256)

    # test full mask
    mask = nn.Transformer.generate_square_subsequent_mask(5, device=device)
    mask2 = MaskGenerationMixin(history_steps=2)._generate_structural_mask(5, device=device)
    mask3 = MaskGenerationMixin()._generate_structural_mask(5, device=device)
    mask4 = MaskGenerationMixin(history_steps=0)._generate_structural_mask(5, device=device)

    assert not torch.allclose(mask, mask2)
    assert torch.allclose(mask, mask3)
    assert not torch.allclose(mask, mask4)

    print("Mask generation successful.")


    # test whether relative position embeddings do not change output values at different positions
    history_steps = 2
    # 0 < history_steps < inf only work for one layer, because higher layer expand the receptive field
    model_test = RoPEDecoderOnly(d_model=16, history_steps=history_steps, num_layers=1).to(device)
    # stop undeterminism by setting model to eval
    model_test.eval()
    data = torch.randn(1, 8 + history_steps * 2, 16).to(device)
    shifted_data = torch.roll(data, shifts=history_steps, dims=1)
    out1 = model_test(data)[:, history_steps:(data.shape[1]-history_steps), :]
    out2 = model_test(shifted_data)[:, (2*history_steps):, :]
    assert torch.allclose(out1, out2, atol=1e-4)
    print("Relative position embeddings test successful.")


    # test whether model not violate causality
    model_causality = RoPEDecoderOnly(d_model=16, history_steps=None, num_layers=1).to(device)
    model_causality.eval()
    seq_len = 10
    data_causality = torch.randn(1, seq_len, 16).to(device)
    out_causality = model_causality(data_causality)
    for t in range(1, seq_len):
        # output at position t should not depend on input at position > t
        input_modified = data_causality.clone()
        input_modified[0, t+1:, :] += 100.0  # large change
        out_modified = model_causality(input_modified)
        assert torch.allclose(out_causality[0, t, :], out_modified[0, t, :], atol=1e-4)
    print("Causality test successful.")
