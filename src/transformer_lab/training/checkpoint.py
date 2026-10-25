import torch


def save_checkpoint(model, optimizer, path):
    """Save model and AdamW state to continue training later (Raschka, ch. 5.4)."""
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, path)


def load_checkpoint(model, optimizer, path, device):
    """Load weights and optimizer state into freshly created model and optimizer (Raschka, ch. 5.4)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
