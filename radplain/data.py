"""Select a bounded experiment with text deduplication before training."""
import argparse
import random
from collections import Counter
from pathlib import Path
from .common import assert_disjoint, fingerprint, load_config, write_json, write_jsonl


def clean_rows(rows):
    result, seen = [], set()
    counts = Counter()
    for index, row in enumerate(rows):
        report = str(row.get("radiology_report") or "").strip()
        target = str(row.get("layman_report") or "").strip()
        if not 20 <= len(report) <= 2400 or not 20 <= len(target) <= 2400:
            counts["empty_or_outside_character_range"] += 1
            continue
        key = fingerprint(report)
        if key in seen:
            counts["duplicate_report"] += 1
            continue
        seen.add(key)
        result.append({"id": key, "source": row.get("source", "unknown"),
                       "original_index": index, "report": report, "reference_en": target})
    return result, dict(counts)


def prepare(config_path, out):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    cfg = load_config(config_path)
    out = Path(out)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Already prepared: {out}; use a new experiment directory")
    revision = HfApi().dataset_info(cfg["dataset"]).sha
    ds = load_dataset(cfg["dataset"], revision=revision)
    train, train_counts = clean_rows(ds["train"])
    held, held_counts = clean_rows(ds["validation"])
    # Exclude ALL official validation report hashes, including length-filtered rows.
    held_hashes = {fingerprint(x["radiology_report"]) for x in ds["validation"]}
    before = len(train)
    train = [x for x in train if x["id"] not in held_hashes]
    random.Random(cfg["seed"]).shuffle(train)
    random.Random(cfg["seed"] + 1).shuffle(held)
    nt, nv, ne = cfg["train_size"], cfg["validation_size"], cfg["test_size"]
    if len(train) < nt or len(held) < nv + ne:
        raise ValueError("Insufficient unique eligible rows; do not silently shrink the experiment")
    splits = {"train": train[:nt], "validation": held[:nv], "test": held[nv:nv + ne]}
    assert_disjoint(splits)
    for name, rows in splits.items():
        write_jsonl(out / f"{name}.jsonl", rows)
    manifest = {"dataset": cfg["dataset"], "revision": revision, "seed": cfg["seed"],
                "official_counts": {k: len(v) for k, v in ds.items()},
                "filters": {"train": train_counts, "validation": held_counts},
                "train_removed_for_validation_overlap": before - len(train),
                "available_unique": {"train": len(train), "held_out": len(held)},
                "counts": {k: len(v) for k, v in splits.items()},
                "source_counts": {k: dict(Counter(r["source"] for r in v)) for k, v in splits.items()},
                "split_definition": "Train from official train; seeded dev and final test subsets from official validation. Official test is not used.",
                "limitations": ["Report-text independence is checked; patient independence cannot be established without patient IDs.",
                                "20-2400 character filtering excludes shortest and longest reports.",
                                "Chinese references will be synthetic and are not clinician-validated."]}
    write_json(out / "manifest.json", manifest)
    print(manifest, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pilot.json")
    p.add_argument("--out", default="data/pilot_v1")
    a = p.parse_args()
    prepare(a.config, a.out)
