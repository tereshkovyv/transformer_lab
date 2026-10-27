import torch
import torch.nn as nn


# Elfwing, Uchibe, Doya, 2017, "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning" (SiLU)
class SiLU(nn.Module):
    """Функция активации SiLU"""

    def forward(self, x):
        return x * torch.sigmoid(x)


# Shazeer, 2020, "GLU Variants Improve Transformer"
class SwiGLUFeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)  # gate
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False)  # up
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False)  # down
        self.silu = SiLU()

    def forward(self, x):
        return self.fc3(self.silu(self.fc1(x)) * self.fc2(x))
