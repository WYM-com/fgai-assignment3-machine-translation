import math

import torch
from torch import nn
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

PAD, BOS, EOS, UNK = 0, 1, 2, 3


def train_tokenizer(rows, vocab_size):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator((r[k] for r in rows for k in ("source", "reference")), trainer)
    return tokenizer


def encode(tokenizer, text, max_length):
    return [BOS] + tokenizer.encode(text).ids[:max_length - 2] + [EOS]


class BasicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, layers=3,
                 feedforward=1024, dropout=0.1, **unused):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.d_model = d_model
        pos = torch.arange(2048).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(2048, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * scale), torch.cos(pos * scale)
        self.register_buffer("position", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
            num_encoder_layers=layers, num_decoder_layers=layers,
            dim_feedforward=feedforward, dropout=dropout, batch_first=True)
        self.output = nn.Linear(d_model, vocab_size)

    def embed(self, ids):
        return self.dropout(self.embedding(ids) * math.sqrt(self.d_model) + self.position[:, :ids.size(1)])

    def forward(self, source, target):
        mask = torch.ones(target.size(1), target.size(1), device=target.device, dtype=torch.bool).triu(1)
        hidden = self.transformer(self.embed(source), self.embed(target), tgt_mask=mask,
            src_key_padding_mask=source.eq(PAD), tgt_key_padding_mask=target.eq(PAD),
            memory_key_padding_mask=source.eq(PAD))
        return self.output(hidden)

    @torch.inference_mode()
    def generate(self, source, max_new_tokens):
        self.eval()
        padding = source.eq(PAD)
        memory = self.transformer.encoder(self.embed(source), src_key_padding_mask=padding)
        target = torch.full((source.size(0), 1), BOS, device=source.device, dtype=torch.long)
        finished = torch.zeros(source.size(0), device=source.device, dtype=torch.bool)
        for _ in range(max_new_tokens):
            mask = torch.ones(target.size(1), target.size(1), device=source.device, dtype=torch.bool).triu(1)
            hidden = self.transformer.decoder(self.embed(target), memory,
                tgt_mask=mask, memory_key_padding_mask=padding)
            logits = self.output(hidden[:, -1])
            logits[:, [PAD, BOS]] = -torch.inf
            next_id = logits.argmax(-1).masked_fill(finished, EOS)
            target = torch.cat([target, next_id[:, None]], dim=1)
            finished |= next_id.eq(EOS)
            if finished.all():
                break
        return target
