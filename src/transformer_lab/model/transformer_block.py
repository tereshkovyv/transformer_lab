import torch.nn as nn

from transformer_lab.model.attention import MultiHeadAttention, MultiHeadAttentionV2
from transformer_lab.model.feed_forward import FeedForward, SwiGLUFeedForward
from transformer_lab.model.layer_norm import LayerNorm, RMSNorm


# Radford et al., 2019, "Language Models are Unsupervised Multitask Learners" (GPT-2: LayerNorm before each sub-block)
class TransformerBlock(nn.Module):
    """Блок трансформера как в GPT-2 и книге Рашки (Raschka, ch. 4.5)."""

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
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


# Touvron et al., 2023, "Llama 2: Open Foundation and Fine-Tuned Chat Models"
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
