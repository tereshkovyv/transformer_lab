import torch
from transformer_lab.model.gpt_model import GPTModel
from transformer_lab.model.generate import generate, generate_text_simple

cfg = {"vocab_size": 50, "context_length": 8, "emb_dim": 16, "n_heads": 4,
       "n_layers": 2, "drop_rate": 0.0, "qkv_bias": False}
torch.manual_seed(123)
model = GPTModel(cfg).eval()

idx = torch.randint(0, 50, (2, 6))
out = generate_text_simple(model, idx, max_new_tokens=5, context_size=8)

# Prompt is kept, exactly max_new_tokens are appended
assert out.shape == (2, 11)
assert torch.equal(out[:, :6], idx)

# First new token is the argmax of the model's last-position logits
with torch.no_grad():
    assert torch.equal(out[:, 6], model(idx)[:, -1].argmax(dim=-1))

# 6 + 5 = 11 > context_length = 8: cropping kept pos_emb in range, otherwise we'd crash above

print("generate_text_simple checks passed")

# temperature = 0 is exactly greedy decoding
assert torch.equal(generate(model, idx, 5, 8), generate_text_simple(model, idx, 5, 8))

# top_k = 1 leaves one candidate, so sampling is forced to be greedy too
assert torch.equal(generate(model, idx, 5, 8, temperature=1.0, top_k=1), generate_text_simple(model, idx, 5, 8))

# With top_k = 5 every sampled first token is among the 5 largest logits (batch of 2 also tests min_val shape)
with torch.no_grad():
    top5 = model(idx)[:, -1].topk(5).indices  # (b, 5)
for seed in range(20):
    torch.manual_seed(seed)
    first = generate(model, idx, 1, 8, temperature=1.0, top_k=5)[:, -1:]
    assert (first == top5).any(dim=-1).all()

# Same seed -> same sample
torch.manual_seed(0); a = generate(model, idx, 5, 8, temperature=1.0)
torch.manual_seed(0); b = generate(model, idx, 5, 8, temperature=1.0)
assert torch.equal(a, b)

print("generate checks passed")
