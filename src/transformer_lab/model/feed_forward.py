import torch
import torch.nn as nn


# Hendrycks, Gimpel, 2016, "Gaussian Error Linear Units (GELUs)"
class GELU(nn.Module):
    """Функция активации GELU"""

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))


# Vaswani et al., 2017, "Attention Is All You Need"
# Raschka, ch. 4.3
class FeedForward(nn.Module):
    """FeedForward слой классического трансформера (Raschka, ch. 4.3)."""

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
