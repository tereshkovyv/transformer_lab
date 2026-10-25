import tiktoken
import torch
from torch.utils.data import DataLoader, Subset
from transformer_lab.model.gpt_model import GPTModel
from transformer_lab.training.data import TokenDataset
from transformer_lab.training.train import train_model_simple

cfg = {"vocab_size": 50257, "context_length": 32, "emb_dim": 32, "n_heads": 4,
       "n_layers": 2, "drop_rate": 0.0, "qkv_bias": False}
torch.manual_seed(123)
model = GPTModel(cfg)
data = Subset(TokenDataset("validation", max_length=32), range(8))  # 8 windows = 2 batches
loader = DataLoader(data, batch_size=4, shuffle=False, drop_last=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

train_losses, val_losses, tokens = train_model_simple(
    model, loader, loader, optimizer, "cpu", num_epochs=60, eval_freq=10, eval_iter=2,
    start_context="The", tokenizer=tiktoken.get_encoding("gpt2"))

# 60 epochs x 2 batches = 120 steps, evaluated at steps 0, 10, ..., 110
assert len(train_losses) == len(val_losses) == len(tokens) == 12
assert tokens[-1] == 111 * 4 * 32  # tokens seen after step 110 (111 steps x batch 4 x 32 tokens)

# The loop actually learns: on 2 fixed batches loss must drop far below ln(V) ≈ 10.8
assert train_losses[-1] < train_losses[0] - 3
assert model.training  # evaluation switched the model back to train mode

print("train_model_simple checks passed")
