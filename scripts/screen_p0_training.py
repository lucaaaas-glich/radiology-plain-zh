"""Run P0 lexical risk diagnostics and screen preserved V1 training records.

Usage, from repository root, without a model or network:
  python -m scripts.screen_p0_training --stage development
  python -m scripts.screen_p0_training --stage freeze
  python -m scripts.screen_p0_training --stage final

The final stage verifies the frozen rules/config/calibration before evaluating
validation; it refuses to overwrite outputs. Raw text stays under ignored data/.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from radplain.common import read_jsonl, write_json, write_jsonl
from radplain.quality_p0 import VERSION, LIMITATIONS, calibration_report, screen_row, summarize_rows


RULE_FILES = ("radplain/quality_p0.py", "scripts/screen_p0_training.py", "radplain/quantity_v2.py",
              "radplain/guard_v2.py", "radplain/fidelity.py", "radplain/common.py", "configs/p0_quality.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_calibration(path, config):
    if digest(path) != config["calibration_sha256"]:
        raise ValueError("Calibration changed since the pre-implementation freeze")
    pairs = read_jsonl(path)
    ids = [p["pair_id"] for p in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate calibration pair IDs")
    dev = {p["family"] for p in pairs if p["split"] == "development"}
    val = {p["family"] for p in pairs if p["split"] == "validation"}
    if dev & val:
        raise ValueError("Calibration family leakage across splits")
    for split in ("development", "validation"):
        if sum(p["split"] == split for p in pairs) != config[split + "_pairs"]:
            raise ValueError("Unexpected calibration split count")
    if len(pairs) != config["development_pairs"] + config["validation_pairs"]:
        raise ValueError("Unexpected calibration split")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "freeze", "final"), required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--public-summary", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "configs/p0_quality.json").read_text())
    input_path = root / (args.input or config["default_input"])
    output = root / (args.output_dir or config["default_output_dir"])
    public = root / (args.public_summary or config["default_public_summary"])
    if not output.resolve().is_relative_to((root / "data").resolve()):
        raise ValueError("Raw diagnostic output must stay under ignored repository data/")
    output.mkdir(parents=True, exist_ok=True)
    calibration_path = root / config["calibration_path"]
    pairs = validate_calibration(calibration_path, config)
    code_hashes = {p: digest(root / p) for p in RULE_FILES}
    lock_path = output / "rules.lock.json"
    if args.stage == "development":
        if lock_path.exists():
            raise FileExistsError("Rules already frozen; development retuning is closed")
        summary, outcomes = calibration_report(pairs, "development")
        write_json(output / "calibration_development.json", {"summary": summary, "outcomes": outcomes,
                   "rule_sha256": code_hashes, "calibration_sha256": digest(calibration_path)})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.stage == "freeze":
        if lock_path.exists():
            raise FileExistsError("Refusing to replace frozen rules")
        development = json.loads((output / "calibration_development.json").read_text())
        if development["rule_sha256"] != code_hashes:
            raise ValueError("Run development diagnostics on the exact candidate before freezing")
        write_json(lock_path, {"version": VERSION, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                   "rule_sha256": code_hashes, "calibration_sha256": digest(calibration_path),
                   "development_diagnostic_sha256": digest(output / "calibration_development.json"),
                   "validation_outputs_read_at_freeze": False,
                   "policy": "No rule tuning after this freeze; report validation errors unchanged."})
        print(json.dumps({"rules_frozen": True, "rule_files": len(code_hashes)}))
        return
    lock = json.loads(lock_path.read_text())
    if lock["rule_sha256"] != code_hashes or lock["calibration_sha256"] != digest(calibration_path):
        raise ValueError("Rules or calibration differ from the pre-validation freeze")
    development_path = output / "calibration_development.json"
    if lock["development_diagnostic_sha256"] != digest(development_path):
        raise ValueError("Development diagnostics changed since freeze")
    outputs = [output / name for name in ("per_sample.jsonl", "risk_details.local.jsonl", "calibration_validation.json")]
    if any(p.exists() for p in outputs) or public.exists():
        raise FileExistsError("Refusing to overwrite a finalized screen or validation result")
    raw_rows = read_jsonl(input_path)
    if len({r["id"] for r in raw_rows}) != len(raw_rows):
        raise ValueError("Duplicate training IDs")
    screened, details = [], []
    for row in raw_rows:
        screened_row, detail = screen_row(row)
        screened.append(screened_row)
        if detail["issues"]:
            details.append(detail)
    val_summary, val_outcomes = calibration_report(pairs, "validation")
    write_jsonl(outputs[0], screened)
    write_jsonl(outputs[1], details)
    write_json(outputs[2], {"summary": val_summary, "outcomes": val_outcomes,
                          "rule_sha256": code_hashes, "rules_lock_sha256": digest(lock_path)})
    for path in outputs:
        path.chmod(0o600)
    summary = {"schema_version": VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "purpose": "CPU lexical training-target risk screening; no clinical correctness determination",
               "input": {"path": str(input_path.relative_to(root)) if input_path.is_relative_to(root) else str(input_path),
                         "sha256": digest(input_path), "all_rows_preserved": len(screened) == len(raw_rows)},
               "source_sha256_definition": "SHA256 of exact report UTF-8; historical id uses normalized text separately",
               "source_length_definition": "Unicode code points", "rules": lock,
               "calibration": {"sha256": digest(calibration_path), "pair_count": len(pairs),
                               "candidate_count": len(pairs) * 2, "split_unit": "family",
                               "label_provenance": "AI_authored_known_synthetic_edit_not_medical_annotation",
                               "human_review_count": 0,
                               "development": json.loads(development_path.read_text())["summary"],
                               "validation": val_summary},
               **summarize_rows(screened), "limitations": LIMITATIONS,
               "local_artifacts": {p.name: {"path": str(p.relative_to(root)), "sha256": digest(p)} for p in outputs}}
    write_json(public, summary)
    print(json.dumps({"counts": summary["counts"], "validation": val_summary["counts"],
                      "public_summary": str(public)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
