import torch
from transformer_lab.model.gpt_model import GPTModel

cfg = {"vocab_size": 50, "context_length": 8, "emb_dim": 16, "n_heads": 4,
       "n_layers": 2, "drop_rate": 0.0, "qkv_bias": False}
torch.manual_seed(123)
model = GPTModel(cfg).eval()

idx = torch.randint(0, 50, (2, 8))
logits = model(idx)
assert logits.shape == (2, 8, 50)

# Causality at the token level: changing later tokens leaves earlier logits unchanged
k = 5
idx_changed = idx.clone()
idx_changed[:, k:] = (idx[:, k:] + 1) % 50
assert torch.allclose(logits[:, :k], model(idx_changed)[:, :k], atol=1e-6)

# Parameter count of GPT_CONFIG_124M matches the book (ch. 4.6): 163,009,536
# The meta device builds shapes only, no memory is allocated
cfg_124m = {"vocab_size": 50257, "context_length": 1024, "emb_dim": 768, "n_heads": 12,
            "n_layers": 12, "drop_rate": 0.1, "qkv_bias": False}
with torch.device("meta"):
    big = GPTModel(cfg_124m)
assert sum(p.numel() for p in big.parameters()) == 163_009_536

print("GPTModel checks passed")
