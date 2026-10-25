import torch


def calc_loss_batch(input_batch, target_batch, model, device):
    """Mean cross-entropy over all positions of a batch (Raschka, ch. 5.1)."""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)  # (b, T, vocab)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Average batch loss over the first num_batches batches (all if None)."""
    num_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    total_loss = 0.0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i == num_batches:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
    return total_loss / num_batches
