import torch


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Простой жадный генератор (Raschka, ch. 4.7)."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # keep only what fits into the context window
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # (b, vocab_size): prediction for the next token
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (b, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # (b, T + 1)
    return idx


# Fan, Lewis, Dauphin, 2018, "Hierarchical Neural Story Generation" (top-k sampling)
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None):
    """Генератор с температурой и top-k sampling (Raschka, ch. 5.3)."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1:]  # (b, 1): the k-th largest logit per row
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (b, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx
