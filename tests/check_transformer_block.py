import torch
from transformer_lab.model.transformer_block import TransformerBlock

cfg = {"emb_dim": 16, "context_length": 8, "n_heads": 4, "drop_rate": 0.0, "qkv_bias": False}
torch.manual_seed(123)
block = TransformerBlock(cfg).eval()

x = torch.randn(2, 8, 16)
out = block(x)
assert out.shape == x.shape

# Causality survives the whole block (LayerNorm and FeedForward are per-token)
k = 5
x_changed = x.clone()
x_changed[:, k:] = torch.randn(2, 8 - k, 16)
assert torch.allclose(out[:, :k], block(x_changed)[:, :k], atol=1e-6)

# Shortcuts: if both sub-blocks output zero, the block is the identity
with torch.no_grad():
    block.att.out_proj.weight.zero_()
    block.att.out_proj.bias.zero_()
    block.ff.layers[2].weight.zero_()
    block.ff.layers[2].bias.zero_()
assert torch.allclose(block(x), x)

print("TransformerBlock checks passed")
