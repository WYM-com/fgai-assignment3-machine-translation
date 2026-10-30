import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def config():
    return read_json(ROOT / "config.json")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def device():
    # CPU fallback is deliberately supported for tests; full training targets CUDA.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_rows(split):
    return read_json(ROOT / "data" / f"{split}.json")


def token_kwargs():
    return {"token": os.environ.get("HF_TOKEN") or None}
