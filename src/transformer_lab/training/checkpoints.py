import os

import torch


def save_progress(path, model, optimizer, scaler, global_step, tokens_seen, history, track_lrs):
    tmp_path = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "history": history,
        "track_lrs": track_lrs,
    }, tmp_path)
    os.replace(tmp_path, path)
    print(f"Progress saved: {path} (step {global_step})")


def load_progress(path, model, optimizer, scaler, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    return state["global_step"], state["tokens_seen"], state["history"], state["track_lrs"]
