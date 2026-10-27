import math

import torch


def calc_loss_batch(input_batch, target_batch, model, device):
    """(Raschka, ch. 5.1)."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())

def calc_loss_loader(data_loader, model, device, num_batches=None):
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    total_loss = 0.0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i == num_batches:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total_loss / num_batches

def train_model_min(model, train_loader, val_loader, optimizer, device,
                    warmup_steps, min_lr, eval_freq, eval_iter):
    peak_lr = optimizer.param_groups[0]["lr"]
    total_steps = len(train_loader)
    scaler = torch.amp.GradScaler("cuda")
    train_losses, val_losses, lrs = [], [], []
    for step, (input_batch, target_batch) in enumerate(train_loader):
        if step < warmup_steps:
            lr = peak_lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = calc_loss_batch(input_batch, target_batch, model, device)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(loss.item())
        lrs.append(lr)

        if (step + 1) % eval_freq == 0 or step + 1 == total_steps:
            model.eval()
            with torch.no_grad():
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
            model.train()
            val_losses.append((step + 1, val_loss))
            print(f"Step {step + 1}: train loss {loss.item():.3f}, val loss {val_loss:.3f}")
    return train_losses, val_losses, lrs
