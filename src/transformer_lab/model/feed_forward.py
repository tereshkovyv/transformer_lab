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


# Elfwing, Uchibe, Doya, 2017, "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning" (SiLU)
class SiLU(nn.Module):
    """Функция активации SiLU (Raschka, ch05/07_gpt_to_llama)."""

    def forward(self, x):
        return x * torch.sigmoid(x)


# Shazeer, 2020, "GLU Variants Improve Transformer"
class SwiGLUFeedForward(nn.Module):
    """FeedForward с гейтом SwiGLU, как в Llama (Raschka, ch05/07_gpt_to_llama)."""

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)  # gate
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)  # up
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False)  # down
        self.silu = SiLU()

    def forward(self, x):
        return self.fc3(self.silu(self.fc1(x)) * self.fc2(x))
