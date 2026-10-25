import tempfile

import numpy as np
import tiktoken
import torch
from torch.utils.data import Dataset
from transformer_lab.training.data import DATA_DIR, TokenDataset, read_split_lines, tokenize_split


# Reference copy of GPTDatasetV1 from Raschka, "Build a Large Language Model (From Scratch)", ch. 2.6
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1:i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# Row counts of the pinned revision
val_lines = read_split_lines("validation")
assert len(val_lines) == 3760

train_lines = read_split_lines("train")
assert len(train_lines) == 1_801_350  # both train shards are read, in order
assert train_lines[0] == "" and train_lines[1].startswith(" = ")  # blank line + article title

print("read_split_lines checks passed")

tokenize_split("validation")
val_ids = np.load(f"{DATA_DIR}/tokens/validation.npy")
assert val_ids.dtype == np.uint16 and val_ids.max() < 50257
assert len(val_ids) == 247_289  # measured earlier on the same text

# BPE is lossless: decoding gives back exactly the original text
tokenizer = tiktoken.get_encoding("gpt2")
assert tokenizer.decode(val_ids.tolist()) == "".join(val_lines)

print("tokenize_split checks passed")

# Same (input, target) pairs as the book's GPTDatasetV1 with stride = max_length
T = 256
ours = TokenDataset("validation", max_length=T)
book = GPTDatasetV1("".join(val_lines), tokenizer, max_length=T, stride=T)
assert len(ours) == len(book) == 965
for i in range(len(ours)):
    x, y = ours[i]
    assert torch.equal(x, book[i][0]) and torch.equal(y, book[i][1])
assert ours[0][0].dtype == torch.int64

# Missing tokens are prepared on first use, giving the same array
with tempfile.TemporaryDirectory() as tmp:
    auto = TokenDataset("validation", max_length=T, data_dir=tmp)
    assert np.array_equal(auto.token_ids, val_ids)

print("TokenDataset checks passed")
