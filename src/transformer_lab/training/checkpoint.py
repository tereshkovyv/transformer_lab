import torch


def save_checkpoint(model, optimizer, path):
    """Save model and AdamW state to continue training later (Raschka, ch. 5.4)."""
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, path)
