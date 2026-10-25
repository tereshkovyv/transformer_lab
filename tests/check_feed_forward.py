import torch
import torch.nn.functional as F
from transformer_lab.model.feed_forward import GELU, FeedForward

torch.manual_seed(123)

# GELU matches PyTorch's tanh approximation
x = torch.linspace(-4, 4, 101)
assert torch.allclose(GELU()(x), F.gelu(x, approximate="tanh"), atol=1e-6)

# FeedForward keeps the shape and acts on each token independently
ff = FeedForward({"emb_dim": 16})
x = torch.randn(2, 5, 16)
out = ff(x)
assert out.shape == x.shape
assert torch.allclose(out[:, 2], ff(x[:, 2]), atol=1e-6)

print("FeedForward checks passed")
