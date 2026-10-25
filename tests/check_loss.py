import math

import torch
from torch.utils.data import DataLoader
from transformer_lab.model.gpt_model import GPTModel
from transformer_lab.training.data import TokenDataset
from transformer_lab.training.train import calc_loss_batch, calc_loss_loader

cfg = {"vocab_size": 50257, "context_length": 32, "emb_dim": 16, "n_heads": 4,
       "n_layers": 2, "drop_rate": 0.0, "qkv_bias": False}
torch.manual_seed(123)
model = GPTModel(cfg).eval()
loader = DataLoader(TokenDataset("validation", max_length=32), batch_size=4, shuffle=False, drop_last=True)
x, y = next(iter(loader))

with torch.no_grad():
    loss = calc_loss_batch(x, y, model, "cpu")

    # Equals the formula written by hand: -mean log p(target)
    log_probs = torch.log_softmax(model(x), dim=-1)
    manual = -log_probs.gather(-1, y.unsqueeze(-1)).mean()
    assert torch.allclose(loss, manual, atol=1e-5)

    # Untrained model is close to uniform guessing: loss ≈ ln(vocab_size) ≈ 10.82
    assert abs(loss.item() - math.log(50257)) < 0.5

    # Loader loss is the mean of its batch losses
    batch_losses = [calc_loss_batch(a, b, model, "cpu").item() for a, b in list(loader)[:3]]
    assert math.isclose(calc_loss_loader(loader, model, "cpu", num_batches=3), sum(batch_losses) / 3)

print("loss checks passed")
