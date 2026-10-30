"""Normalize, filter and de-duplicate before a fixed, disjoint random split."""
import random
import re
import unicodedata

from datasets import load_dataset
from huggingface_hub import HfApi

from .common import ROOT, config, digest, write_json


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def clean_pairs(rows, max_words):
    cleaned, seen_en, seen_fr = [], set(), set()
    for row in rows:
        en = normalize(row["translation"]["en"])
        fr = normalize(row["translation"]["fr"])
        a, b = len(en.split()), len(fr.split())
        if not (2 <= a <= max_words and 2 <= b <= max_words):
            continue
        if max(a, b) / min(a, b) > 3:
            continue
        # Avoid repeated source OR target sentences crossing split boundaries.
        en_key, fr_key = en.casefold(), fr.casefold()
        if en_key in seen_en or fr_key in seen_fr:
            continue
        seen_en.add(en_key)
        seen_fr.add(fr_key)
        cleaned.append({"id": digest([en, fr])[:20], "source": en, "reference": fr})
    return cleaned


def prepare():
    cfg = config()
    revision = HfApi().dataset_info(cfg["dataset"]).sha
    raw = load_dataset(cfg["dataset"], cfg["dataset_config"], revision=revision, split="train")
    rows = clean_pairs(raw, cfg["max_words"])
    random.Random(cfg["seed"]).shuffle(rows)
    sizes = [cfg[k] for k in ("train_size", "validation_size", "test_size")]
    if len(rows) < sum(sizes):
        raise ValueError(f"Only {len(rows)} eligible rows; requested {sum(sizes)}")
    offset, splits = 0, {}
    for name, size in zip(("train", "validation", "test"), sizes):
        splits[name] = rows[offset:offset + size]
        write_json(ROOT / "data" / f"{name}.json", splits[name])
        offset += size
    stats = {
        "dataset": cfg["dataset"], "configuration": cfg["dataset_config"],
        "revision": revision, "raw_rows": len(raw), "eligible_rows": len(rows),
        "seed": cfg["seed"], "config_hash": digest(cfg),
        "splits": {name: {"count": len(value), "sha256": digest(value),
            "mean_source_words": sum(len(r["source"].split()) for r in value) / len(value),
            "mean_target_words": sum(len(r["reference"].split()) for r in value) / len(value)}
            for name, value in splits.items()},
        "limitation": "Sentence-level split; book-level grouping unavailable in this pipeline. Related passages and pretraining contamination remain possible."
    }
    write_json(ROOT / "results" / "dataset.json", stats)
    print(stats)


if __name__ == "__main__":
    prepare()
