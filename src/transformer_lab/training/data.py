import os

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

# Merity et al., 2016, "Pointer Sentinel Mixture Models" (WikiText-103)
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
# Touvron et al., 2023, "Llama 2: Open Foundation and Fine-Tuned Chat Models" (SentencePiece BPE, 32 000 tokens)
LLAMA2_REVISION = "d02ad6cb9dd2c2296a6332199fa2fdca5938fef0"


class Llama2Tokenizer:
    def __init__(self):
        path = hf_hub_download("hf-internal-testing/llama-tokenizer", "tokenizer.model", revision=LLAMA2_REVISION)
        self.sp = spm.SentencePieceProcessor(model_file=path)

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, ids):
        return self.sp.decode(ids)


def read_lines(shard):
    path = hf_hub_download("Salesforce/wikitext",
                           f"wikitext-103-raw-v1/{shard}.parquet",
                           repo_type="dataset",
                           revision=WIKITEXT_REVISION)
    return pq.read_table(path)["text"].to_pylist()


def tokenize_split(split):
    if split == "train":
        lines = read_lines("train-00000-of-00002") + read_lines("train-00001-of-00002")
    else:
        lines = read_lines("validation-00000-of-00001")
    chunks = ["".join(lines[i:i + 10_000]) for i in range(0, len(lines), 10_000)]
    sp = Llama2Tokenizer().sp
    ids = sp.encode(chunks[:1]) + [x[2:] for x in sp.encode(["\n" + chunk for chunk in chunks[1:]])]
    return np.concatenate([np.array(x, dtype=np.uint16) for x in ids])


class TokenDataset(Dataset):
    def __init__(self, split, max_length, data_dir):
        path = f"{data_dir}/tokens/llama2/{split}.npy"
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, tokenize_split(split))
        self.token_ids = np.load(path)
        self.max_length = max_length

    def __len__(self):
        return (len(self.token_ids) - 1) // self.max_length

    def __getitem__(self, idx):
        start = idx * self.max_length
        chunk = torch.from_numpy(self.token_ids[start:start + self.max_length + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]
