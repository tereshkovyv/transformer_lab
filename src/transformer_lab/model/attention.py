import torch
import torch.nn as nn

# Vaswani et al., 2017, "Attention Is All You Need"
# Raschka, ch. 3.6
class MultiHeadAttention(nn.Module):
    """Классический блок внимания трансформера. Causal multi-head self-attention (Raschka гл. 3.6)"""
    """Оригинальная версия из статьи Attention Is All You Need"""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # Upper triangle above the diagonal = future positions to hide
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape

        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ values).transpose(1, 2)  # (b, T, H, head_dim)
        context = context.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)


# Su et al., 2021, "RoFormer: Enhanced Transformer with Rotary Position Embedding"
# То есть, классический RoPE. Код из Raschka, ch. 5.3
def precompute_rope_params(head_dim, context_length, theta_base=10_000):
    """cos/sin для RoPE (Raschka, ch05/07_gpt_to_llama)."""
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.arange(context_length)[:, None] * inv_freq[None, :]  # (T, head_dim/2)
    angles = torch.cat([angles, angles], dim=1)  # (T, head_dim)
    return torch.cos(angles), torch.sin(angles)


def compute_rope(x, cos, sin):
    """Поворот пар (x1, x2)"""
    seq_len, half = x.shape[2], x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos[:seq_len] + rotated * sin[:seq_len]


# Touvron et al., 2023, "Llama 2: Open Foundation and Fine-Tuned Chat Models"
class MultiHeadAttentionV2(nn.Module):
    """Causal multi-head self-attention с RoPE и out_proj без bias, как в Llama (Raschka, ch05/07_gpt_to_llama)."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.linear_layer = nn.Linear(d_out, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        cos, sin = precompute_rope_params(self.head_dim, context_length)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        b, num_tokens, _ = x.shape

        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Rope
        queries = compute_rope(queries, self.cos, self.sin)
        keys = compute_rope(keys, self.cos, self.sin)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ values).transpose(1, 2)
        context = context.contiguous().view(b, num_tokens, self.d_out)
        return self.linear_layer(context)
