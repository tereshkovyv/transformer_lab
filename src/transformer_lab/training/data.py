import os

import numpy as np
import pandas as pd
import sentencepiece as spm
import tiktoken
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

# Merity et al., 2016, "Pointer Sentinel Mixture Models" (WikiText-103)
REPO_ID = "Salesforce/wikitext"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"  # pinned commit, 2024-01-04
DATA_DIR = "data/wikitext-103"  # default location; every function takes data_dir to override it
SHARDS = {
    "train": ["train-00000-of-00002", "train-00001-of-00002"],
    "validation": ["validation-00000-of-00001"],
}

# Touvron et al., 2023, "Llama 2: Open Foundation and Fine-Tuned Chat Models" (SentencePiece BPE, 32 000 tokens)
# Open copy of Meta's tokenizer.model, sha256 9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347
LLAMA2_REPO_ID = "hf-internal-testing/llama-tokenizer"
LLAMA2_REVISION = "d02ad6cb9dd2c2296a6332199fa2fdca5938fef0"  # pinned commit, 2024-07-29


def read_split_lines(split, data_dir=DATA_DIR):
    """Return the raw lines of a WikiText-103 split, downloading the pinned files if missing."""
    lines = []
    for shard in SHARDS[split]:
        path = hf_hub_download(
            REPO_ID, f"wikitext-103-raw-v1/{shard}.parquet",
            repo_type="dataset", revision=REVISION, local_dir=data_dir,
        )
        lines.extend(pd.read_parquet(path)["text"])
    return lines


class Llama2Tokenizer:
    """Токенизатор Llama 2"""

    def __init__(self, data_dir=DATA_DIR):
        path = hf_hub_download(LLAMA2_REPO_ID, "tokenizer.model", revision=LLAMA2_REVISION,
                               local_dir=f"{data_dir}/tokenizers/llama2")
        self.sp = spm.SentencePieceProcessor(model_file=path)

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, ids):
        return self.sp.decode(ids)

    def encode_lines(self, lines, lines_per_chunk=10_000):
        assert all(line.endswith("\n") for line in lines if line)
        chunks = ["".join(lines[i:i + lines_per_chunk]) for i in range(0, len(lines), lines_per_chunk)]
        token_ids = [np.array(self.sp.encode(chunks[0]), dtype=np.uint16)]
        n_threads = os.cpu_count() or 1
        for i in range(1, len(chunks), n_threads):
            prefixed = ["\n" + chunk for chunk in chunks[i:i + n_threads]]
            for ids in self.sp.encode(prefixed, num_threads=n_threads):
                token_ids.append(np.array(ids[2:], dtype=np.uint16))
        return np.concatenate(token_ids)

def tokens_path(split, data_dir=DATA_DIR, tokenizer="gpt2"):
    # GPT-2 tokens stay where they always were, so existing caches remain valid
    folder = f"{data_dir}/tokens" if tokenizer == "gpt2" else f"{data_dir}/tokens/{tokenizer}"
    return f"{folder}/{split}.npy"


def tokenize_split(split, data_dir=DATA_DIR, tokenizer="gpt2"):
    """Tokenize a whole split and save it as a flat uint16 array (Raschka, ch. 2.5)."""
    lines = read_split_lines(split, data_dir)
    if tokenizer == "gpt2":
        token_ids = tiktoken.get_encoding("gpt2").encode_ordinary("".join(lines))
    else:
        token_ids = Llama2Tokenizer(data_dir).encode_lines(lines)
    path = tokens_path(split, data_dir, tokenizer)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.asarray(token_ids, dtype=np.uint16))


class TokenDataset(Dataset):
    """Датасет (Raschka, ch. 2.6)."""

    def __init__(self, split, max_length, data_dir=DATA_DIR, tokenizer="gpt2"):
        path = tokens_path(split, data_dir, tokenizer)
        if not os.path.exists(path):
            tokenize_split(split, data_dir, tokenizer)
        self.token_ids = np.load(path)
        self.max_length = max_length

    def __len__(self):
        return (len(self.token_ids) - 1) // self.max_length

    def __getitem__(self, idx):
        start = idx * self.max_length
        chunk = torch.from_numpy(self.token_ids[start:start + self.max_length + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]
