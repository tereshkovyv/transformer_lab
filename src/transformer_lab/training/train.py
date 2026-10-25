import torch

from transformer_lab.model.generate import generate_text_simple
from transformer_lab.training.loss import calc_loss_batch, calc_loss_loader


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """Train and val loss on the first eval_iter batches, without dropout or gradients (Raschka, ch. 5.2)."""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """Print a 50-token greedy continuation of start_context."""
    model.eval()
    encoded = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0).to(device)
    token_ids = generate_text_simple(model, encoded, max_new_tokens=50,
                                     context_size=model.pos_emb.weight.shape[0])
    print(tokenizer.decode(token_ids.squeeze(0).tolist()).replace("\n", " "))
    model.train()


# Loshchilov, Hutter, 2019, "Decoupled Weight Decay Regularization" (AdamW, the optimizer passed in)
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """Plain training loop: AdamW steps, periodic evaluation, a sample after each epoch (Raschka, ch. 5.2)."""
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
