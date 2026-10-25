import torch
from transformer_lab.model.attention import MultiHeadAttention

torch.manual_seed(123)
mha = MultiHeadAttention(d_in=16, d_out=16, context_length=8, dropout=0.0, num_heads=4).eval()

x = torch.randn(2, 8, 16)
out = mha(x)
assert out.shape == (2, 8, 16)

# Causality: changing tokens from position k on must not affect outputs before k
k = 5
x_changed = x.clone()
x_changed[:, k:] = torch.randn(2, 8 - k, 16)
out_changed = mha(x_changed)
assert torch.allclose(out[:, :k], out_changed[:, :k])
assert not torch.allclose(out[:, k:], out_changed[:, k:])

print("MultiHeadAttention checks passed")
