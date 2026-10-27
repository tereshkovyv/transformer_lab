import torch.nn as nn

from transformer_lab.model.layer_norm import RMSNorm
from transformer_lab.model.transformer_block import TransformerBlockV2


class GPTModelV2(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlockV2(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.apply(self._init_weights)
        self.out_head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, in_idx):
        x = self.drop_emb(self.tok_emb(in_idx))
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)  # (b, T, vocab_size)
