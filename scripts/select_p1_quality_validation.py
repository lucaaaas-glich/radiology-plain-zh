"""Freeze a metadata-stratified audit sample of old accepted training targets.

Selection accesses source/target bytes only for integrity hashing. It does not
inspect new screening code/results or produce correctness judgments.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from scripts.build_exposure_manifest import fingerprint


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root
    paths = {
        "training": "data/pilot_v1/train.zh.jsonl",
        "old_screen": "data/p0/pre-gpu-20261116/quality/per_sample.jsonl",
        "old18_manifest": "examples/p0/real_target_audit_manifest.json",
        "protocol": "examples/p1/quality_validation_protocol_20261117.json",
    }
    packet_path = root / "data/p1/quality-validation-20261117/real_packet.jsonl"
    manifest_path = root / "examples/p1/real_quality_validation_manifest_20261117.json"
    if packet_path.exists() or manifest_path.exists():
        parser.error("Refusing to overwrite an existing frozen sample")
    protocol = json.loads((root / paths["protocol"]).read_text())
    plan = protocol["real_target_sampling"]
    training = rows(root / paths["training"])
    accepted = [row for row in training if row.get("teacher_accepted") is True]
    if len(accepted) != plan["expected_accepted_rows"]:
        raise ValueError("Accepted original training count mismatch")
    previous = json.loads((root / paths["old18_manifest"]).read_text())["cases"]
    old_ids = {row["id"] for row in previous}
    old_hashes = {row["normalized_source_sha256"] for row in previous}
    if len(old_ids) != plan["exclude_prior_ids"]:
        raise ValueError("Previous audit exclusion count mismatch")
    old_screen = rows(root / paths["old_screen"])
    metadata = {row["id"]: row for row in old_screen}
    if len(metadata) != len(old_screen):
        raise ValueError("Old screening metadata has duplicate IDs")
    strata = defaultdict(list)
    for row in accepted:
        source_hash = fingerprint(row["report"])
        if row["id"] in old_ids or source_hash in old_hashes:
            continue
        meta = metadata[row["id"]]
        if (meta["normalized_source_sha256"] != source_hash or
                meta["features"]["source"] != row["source"]):
            raise ValueError("Original source/old metadata mismatch")
        key = (row["source"], meta["features"]["length_bin"], bool(meta["flags"]))
        strata[key].append(row)
    rng = random.Random(plan["seed"])
    allocation = {}
    chosen = []
    for source in plan["sources"]:
        keys = sorted((key for key in strata if key[0] == source),
                      key=lambda key: (-len(strata[key]), key))
        quota = plan["source_quota"]
        if len(keys) > quota or sum(len(strata[key]) for key in keys) < quota:
            raise ValueError("Source quota cannot cover its nonempty strata")
        allocation.update({key: 1 for key in keys})
        while sum(allocation[key] for key in keys) < quota:
            for key in keys:
                if allocation[key] < len(strata[key]):
                    allocation[key] += 1
                    if sum(allocation[k] for k in keys) == quota:
                        break
        for key in keys:
            chosen.extend(rng.sample(sorted(strata[key], key=lambda row: row["id"]), allocation[key]))
    rng.shuffle(chosen)
    if len(chosen) != plan["sample_size"] or len({row["id"] for row in chosen}) != len(chosen):
        raise ValueError("Sample size or uniqueness mismatch")
    packet = []
    for row in chosen:
        meta = metadata[row["id"]]
        packet.append({
            "id": row["id"], "source": row["source"],
            "length_bin": meta["features"]["length_bin"],
            "old_has_rule_flag": bool(meta["flags"]), "old_flags": meta["flags"],
            "normalized_source_sha256": fingerprint(row["report"]),
            "target_sha256": hashlib.sha256(row["reference_zh"].encode()).hexdigest(),
            "source_characters": len(row["report"]), "target_characters": len(row["reference_zh"]),
            "report": row["report"], "reference_zh": row["reference_zh"],
        })
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet), encoding="utf-8")
    manifest = {
        "version": "p1-real-quality-validation-20261117-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Prospective validation of training-target screening, not model final evaluation",
        "selection_completed_before_semantic_review": True,
        "new_screening_code_or_results_read": False,
        "reviewer_type": "project_AI_not_independent_clinical", "human_reviews": 0,
        "accepted_training_rows": len(accepted), "eligible_after_old18_exclusion": sum(map(len, strata.values())),
        "seed": plan["seed"], "sample_size": len(packet),
        "prior_audit_id_overlap": 0, "prior_audit_source_hash_overlap": 0,
        "inputs_sha256": {path: sha(root / path) for path in paths.values()},
        "selector_sha256": sha(Path(__file__)),
        "packet_path": packet_path.relative_to(root).as_posix(), "packet_sha256": sha(packet_path),
        "stratum_counts": [
            {"source": key[0], "length_bin": key[1], "old_has_rule_flag": key[2],
             "eligible": len(strata[key]), "selected": allocation.get(key, 0)} for key in sorted(strata)],
        "source_counts": dict(Counter(row["source"] for row in packet)),
        "old_flag_presence_counts": dict(Counter("has_flag" if row["old_has_rule_flag"] else "no_flag" for row in packet)),
        "cases": [{key: value for key, value in row.items() if key not in {"report", "reference_zh"}}
                  for row in packet],
        "limits": [
            "Equal source quotas and coverage of rare strata are purposive, not a population-weighted random sample.",
            "Only old P0 flags were used for stratification; they are not correctness labels.",
            "This pool belongs to historical training; it does not establish model evaluation independence.",
            "No human or independent clinical review has been completed by this selection step.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "sample_size", "eligible_after_old18_exclusion", "source_counts", "old_flag_presence_counts", "packet_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
