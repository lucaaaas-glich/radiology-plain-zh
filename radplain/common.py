"""Small deterministic I/O helpers; no credentials or external services."""
import hashlib
import json
import os
import random
import re
from pathlib import Path


def fingerprint(text):
    text = re.sub(r"\s+", " ", str(text).casefold()).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def chinese_ratio(text):
    letters = [c for c in text if c.isalpha()]
    return sum("\u4e00" <= c <= "\u9fff" for c in letters) / max(1, len(letters))


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def assert_disjoint(splits):
    seen = {}
    for split, rows in splits.items():
        for row in rows:
            key = fingerprint(row["report"])
            if key in seen:
                raise ValueError(f"Duplicate report in {seen[key]} and {split}: {key}")
            seen[key] = split
