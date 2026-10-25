import os

import numpy as np
import pandas as pd
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


def tokenize_split(split, data_dir=DATA_DIR):
    """Tokenize a whole split with GPT-2 BPE and save it as a flat uint16 array (Raschka, ch. 2.5)."""
    text = "".join(read_split_lines(split, data_dir))
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = np.array(tokenizer.encode_ordinary(text), dtype=np.uint16)
    os.makedirs(f"{data_dir}/tokens", exist_ok=True)
    np.save(f"{data_dir}/tokens/{split}.npy", token_ids)


class TokenDataset(Dataset):
    """Датасет (Raschka, ch. 2.6)."""

    def __init__(self, split, max_length, data_dir=DATA_DIR):
        path = f"{data_dir}/tokens/{split}.npy"
        if not os.path.exists(path):
            tokenize_split(split, data_dir)
        self.token_ids = np.load(path)
        self.max_length = max_length

    def __len__(self):
        return (len(self.token_ids) - 1) // self.max_length

    def __getitem__(self, idx):
        start = idx * self.max_length
        chunk = torch.from_numpy(self.token_ids[start:start + self.max_length + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]
