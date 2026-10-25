import torch
import torch.nn as nn


# Ba, Kiros, Hinton, 2016, "Layer Normalization"
class LayerNorm(nn.Module):
    """Нормализация по признаку (Raschka, ch. 4.2)."""

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# Zhang, Sennrich, 2019, "Root Mean Square Layer Normalization"
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
