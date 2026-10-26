import math
import os

import torch
from transformer_lab.model.generate import generate_text_simple
from transformer_lab.training.checkpoints import load_progress, save_progress

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

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """Выводит текст после каждой эпохи. Raschka, ch. 5.2."""
    model.eval()
    encoded = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0).to(device)
    token_ids = generate_text_simple(model, encoded, max_new_tokens=50,
                                     context_size=model.trf_blocks[0].att.mask.shape[0])
    print(tokenizer.decode(token_ids.squeeze(0).tolist()).replace("\n", " "))
    model.train()

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def train_model(model, train_loader, val_loader, optimizer, device, n_epochs,
                eval_freq, eval_iter, start_context, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6, use_amp=False,
                log_freq=50, checkpoint_path=None, checkpoint_freq=500):
    """Обучение с warmup, cosine decay и gradient clipping (Raschka, appendix D).

    Логи как у Дениса, шаги считаются с 1:
    - train loss — loss текущего шага, на шаге 1, каждые log_freq шагов и на последнем;
    - val loss (первые eval_iter batch'ей) — до обучения (шаг 0), на шаге 1, каждые eval_freq шагов и на последнем.
    Если задан checkpoint_path, прогресс сохраняется каждые checkpoint_freq шагов,
    а при наличии файла обучение продолжается с него (см. checkpoints.py).
    """
    history = {"step": [], "tokens_seen": [], "train_loss": [], "val_loss": []}
    track_lrs = []
    tokens_seen, start_step = 0, 0

    # Read before load_progress: the loaded optimizer state holds the last lr, not the peak one
    peak_lr = optimizer.param_groups[0]["lr"]
    total_training_steps = len(train_loader) * n_epochs
    lr_increment = (peak_lr - initial_lr) / warmup_steps
    # fp16 gradients can underflow, so the loss is scaled up before backward and unscaled before clipping
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        start_step, tokens_seen, history, track_lrs = load_progress(
            checkpoint_path, model, optimizer, scaler, device)
        print(f"Resuming after step {start_step}")
    else:
        train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
        for key, value in zip(history, (0, 0, train_loss, val_loss)):
            history[key].append(value)
        print(f"Step 000000: Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

    global_step = 0
    for epoch in range(n_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            global_step += 1
            if global_step <= start_step:
                continue  # done before the checkpoint; the loader order is fixed, so these are the same batches
            optimizer.zero_grad()

            step_index = global_step - 1  # the lr schedule counts from 0
            if step_index < warmup_steps:
                lr = initial_lr + step_index * lr_increment  # linear warmup
            else:
                progress = (step_index - warmup_steps) / (total_training_steps - warmup_steps)
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = calc_loss_batch(input_batch, target_batch, model, device)
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            tokens_seen += input_batch.numel()

            is_last = global_step == total_training_steps
            log_train = global_step == 1 or global_step % log_freq == 0 or is_last
            log_val = global_step == 1 or global_step % eval_freq == 0 or is_last
            if log_train or log_val:
                val_loss = None
                if log_val:
                    model.eval()
                    with torch.no_grad():
                        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                    model.train()
                for key, value in zip(history, (global_step, tokens_seen, loss.item(), val_loss)):
                    history[key].append(value)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): Train loss {loss.item():.3f}"
                      + (f", Val loss {val_loss:.3f}" if val_loss is not None else ""))

            if checkpoint_path is not None and global_step % checkpoint_freq == 0 and not is_last:
                save_progress(checkpoint_path, model, optimizer, scaler,
                              global_step, tokens_seen, history, track_lrs)

        generate_and_print_sample(model, tokenizer, device, start_context)

    return history, track_lrs
