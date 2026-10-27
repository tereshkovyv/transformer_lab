import torch.nn as nn

from transformer_lab.model.attention import MultiHeadAttentionV2
from transformer_lab.model.feed_forward import SwiGLUFeedForward
from transformer_lab.model.layer_norm import RMSNorm


class TransformerBlockV2(nn.Module):
    """Блок трансформера как в Llama: RoPE, RMSNorm, SwiGLU (Raschka, ch05/07_gpt_to_llama)."""

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttentionV2(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = SwiGLUFeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # He et al., 2016, "Deep Residual Learning for Image Recognition" (shortcut connections)
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
