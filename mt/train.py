import argparse
import math
import time

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Adafactor
from huggingface_hub import HfApi

from .basic import BasicTransformer, PAD, encode, train_tokenizer
from .common import ROOT, config, device, digest, read_rows, seed_all, sync, token_kwargs, write_json


def train(name):
    cfg = config()
    seed_all(cfg["seed"])
    dev = device()
    train_rows, val_rows = read_rows("train"), read_rows("validation")
    directory = ROOT / "checkpoints" / name
    if directory.exists():
        raise RuntimeError(f"{directory} already exists. Use the saved model or remove its folder explicitly to retrain.")
    directory.mkdir(parents=True)
    opts = cfg[name]
    revision = None
    if name == "basic":
        tokenizer = train_tokenizer(train_rows, cfg["vocab_size"])
        tokenizer.save(str(directory / "tokenizer.json"))
        model = BasicTransformer(tokenizer.get_vocab_size(), **opts).to(dev)
        def tokenize(rows):
            return [(torch.tensor(encode(tokenizer, r["source"], cfg["max_length"])),
                     torch.tensor(encode(tokenizer, r["reference"], cfg["max_length"]))) for r in rows]
        def collate(batch):
            src, tgt = zip(*batch)
            return {"source": pad_sequence(src, batch_first=True, padding_value=PAD),
                    "target": pad_sequence(tgt, batch_first=True, padding_value=PAD)}
        optimizer = torch.optim.AdamW(model.parameters(), lr=opts["learning_rate"], weight_decay=0.01)
    else:
        revision = HfApi().model_info(opts["id"], **token_kwargs()).sha
        tokenizer = AutoTokenizer.from_pretrained(opts["id"], revision=revision, **token_kwargs())
        # Float32 prevents the fp16 overflow frequently encountered in mT5 fine-tuning.
        model = AutoModelForSeq2SeqLM.from_pretrained(opts["id"], revision=revision,
            torch_dtype=torch.float32, **token_kwargs()).to(dev)
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        def tokenize(rows):
            result = []
            for row in rows:
                item = tokenizer("translate English to French: " + row["source"],
                    max_length=cfg["max_length"], truncation=True)
                item["labels"] = tokenizer(text_target=row["reference"],
                    max_length=cfg["max_length"], truncation=True)["input_ids"]
                result.append(item)
            return result
        collate = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
        optimizer = Adafactor(model.parameters(), lr=opts["learning_rate"],
            scale_parameter=False, relative_step=False, warmup_init=False)
    loaders = [DataLoader(tokenize(rows), batch_size=opts["batch_size"],
        shuffle=shuffle, collate_fn=collate) for rows, shuffle in [(train_rows, True), (val_rows, False)]]
    accumulation = opts.get("accumulation", 1)
    total_steps = opts["epochs"] * math.ceil(len(loaders[0]) / accumulation)
    warmup = max(1, int(total_steps * 0.1))
    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    def loss_for(batch):
        batch = {k: v.to(dev) for k, v in batch.items()}
        if name == "basic":
            labels = batch["target"][:, 1:]
            logits = model(batch["source"], batch["target"][:, :-1])
            return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=PAD), int(labels.ne(PAD).sum())
        return model(**batch).loss, int(batch["labels"].ne(-100).sum())

    history, best = [], float("inf")
    sync()
    started = time.perf_counter()
    for epoch in range(opts["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_sum, train_tokens = 0.0, 0
        for index, batch in enumerate(loaders[0]):
            loss, tokens = loss_for(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {name} loss at epoch {epoch + 1}, batch {index}")
            window_start = (index // accumulation) * accumulation
            window_size = min(accumulation, len(loaders[0]) - window_start)
            (loss / window_size).backward()
            train_sum += loss.item() * tokens
            train_tokens += tokens
            if (index + 1) % accumulation == 0 or index + 1 == len(loaders[0]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        val_sum, val_tokens = 0.0, 0
        with torch.inference_mode():
            for batch in loaders[1]:
                loss, tokens = loss_for(batch)
                val_sum += loss.item() * tokens
                val_tokens += tokens
        val_loss = val_sum / val_tokens
        record = {"epoch": epoch + 1, "train_loss": train_sum / train_tokens, "validation_loss": val_loss}
        history.append(record)
        print(record, flush=True)
        if val_loss < best:
            best = val_loss
            if name == "basic":
                torch.save(model.state_dict(), directory / "model.pt")
            else:
                model.save_pretrained(directory)
                tokenizer.save_pretrained(directory)
    sync()
    metadata = {"model": name, "model_id": opts.get("id", "custom-basic-transformer"),
        "revision": revision, "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "training_seconds": time.perf_counter() - started,
        "timing_scope": "All epochs, validation and checkpoint writes; excludes download and tokenization",
        "history": history, "best_validation_loss": best,
        "config_hash": digest(cfg), "train_hash": digest(train_rows), "validation_hash": digest(val_rows),
        "device": str(dev), "dtype": str(next(model.parameters()).dtype)}
    write_json(ROOT / "results" / f"{name}_training.json", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["basic", "mt5"])
    train(parser.parse_args().model)
