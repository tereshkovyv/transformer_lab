import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Нормализация по RMS, без центрирования и сдвига (Raschka, ch05/07_gpt_to_llama)."""

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(mean_sq + self.eps)
        return self.scale * norm_x
