import torch
from transformer_lab.model.layer_norm import LayerNorm

torch.manual_seed(123)
x = torch.randn(2, 5, 16) * 3 + 7
ln = LayerNorm(16)
out = ln(x)

# Per-token mean 0 and variance 1 (scale=1, shift=0 at init)
assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 5), atol=1e-5)
assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(2, 5), atol=1e-3)

# Same result as PyTorch's reference implementation
assert torch.allclose(out, torch.nn.LayerNorm(16, eps=1e-5)(x), atol=1e-5)

print("LayerNorm checks passed")
