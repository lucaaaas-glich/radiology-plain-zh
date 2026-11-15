"""Inventory existing V1--V3 exposure locally, without exporting report text.

Only the standard library is used. No models, subprocesses, network, or source
files are executed. A historical freeze's outputs_seen flag is not evidence of
current non-exposure. Template/near-duplicate independence remains unaudited.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


DATASET_REVISION = "6783ab38f9de28c4372799346f51d3927ae7eb9a"
PARQUET = "data/v2/source/data/validation-00000-of-00001.parquet"
PARQUET_SHA256 = "680bd10de398efbbe7cc115aa861bc12f722fcbcd307cd9c3deedacca8941fce"
HISTORICAL_DIRS = (
    "data/pilot_v1", "data/v2", "data/v3", "examples/v2", "examples/v3",
    "results/pilot_v1", "results/mac_4bit_v1", "results/v2", "results/v3",
)
CANONICAL = {
    "data/pilot_v1/train.jsonl": ("v1_training", "real_report", 4096),
    "data/pilot_v1/validation.jsonl": ("v1_development", "real_report", 256),
    "data/pilot_v1/test.jsonl": ("v1_final_evaluation", "real_report", 256),
    "examples/challenge.jsonl": ("v1_challenge_then_v2_development", "synthetic_report", 20),
    "data/v2/real/dev.jsonl": ("v2_development", "real_report", 16),
    "data/v2/real/holdout.jsonl": ("v2_final_then_v3_development", "real_report", 24),
    "examples/v2/holdout.jsonl": ("v2_final_then_v3_development", "synthetic_report", 48),
    "data/v3/real/holdout.jsonl": ("v3_candidate_reservation", "real_report", 40),
    "data/v3/real/main20.jsonl": ("v3_primary_selection", "real_report", 20),
    "data/v3/development/dev5-old4.jsonl": ("v3_development_selection", "real_report", 4),
    "data/v2/thinking_dev_pairs.jsonl": ("v2_critic_diagnostic", "diagnostic_fragment", 38),
}
FINAL_ARMS = {
    "v2": ("results/v3/real-data/collected/v2/outputs.jsonl", 20),
    "v3": ("results/v3/real-data/collected/v3/outputs.jsonl", 20),
    "without_ledger": ("results/v3/real-data/collected/without_ledger/outputs.jsonl", 10),
}
REQUIRED_JSONL = {
    *(f"data/pilot_v1/{split}.{suffix}.jsonl"
      for split in ("train", "validation", "test") for suffix in ("zh", "teacher-journal")),
    *(f"results/pilot_v1/{arm}/predictions.jsonl"
      for arm in ("baseline", "lora", "challenge-baseline", "challenge-lora")),
    "results/mac_4bit_v1/predictions.jsonl",
    *(f"results/v2/{prefix}{arm}/outputs.jsonl" for prefix in ("", "real-data/")
      for arm in ("final_v1_direct", "final_local_direct", "final_local_evidence",
                  "final_gpu_evidence", "final_gpu_recheck")),
    *(path for path, _ in FINAL_ARMS.values()),
    *(f"results/v3/real-data/{probe}/outputs.jsonl" for probe in (
        "gpu-dev1", "gpu-dev4", "gpu-dev6", "local-dev1", "local-dev2checked",
        "local-dev5", "kernel-probe", "compile-probe")),
    "results/v2/real-data/dev_thinking_critic_01/direct.jsonl",
    "results/v2/real-data/dev_thinking_critic_01/thinking.jsonl",
    "results/v2/dev_critic_qwen35_01/outputs.jsonl",
    "results/v2/dev_critic_qwen35_02/outputs.jsonl",
    "results/v2/dev_nllb600_probe/outputs.jsonl",
    "results/v2/real-data/dev_nllb600_probe/outputs.jsonl",
}
JSON_SOURCES = {
    "results/pilot_v1/case-review.json": (None, "report", "review_record_present", "real_report"),
    "results/v2/guard_diagnostic_dev2.json": ("pairs", "source", "local_diagnostic_result_present", "diagnostic_fragment"),
    "results/v2/guard_diagnostic_dev3.json": ("pairs", "source", "local_diagnostic_result_present", "diagnostic_fragment"),
    "results/v2/dev_nllb600_probe/positive_controls.json": ("cases", "source", "local_output_present", "synthetic_probe"),
}
ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\Z")


def normalize(text):
    """Identical to radplain.common.fingerprint's normalization."""
    return re.sub(r"\s+", " ", str(text).casefold()).strip()


def fingerprint(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def row_fingerprint(row):
    # Same serialization as scripts.summarize_v2.fingerprint, for review links.
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class Inventory:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.records = {}
        self.files = {}
        self.issues = []
        self.cohorts = defaultdict(list)
        self.output_rows = defaultdict(set)
        self.file_report_hashes = defaultdict(list)
        self.ids = defaultdict(set)
        self.manifests = {}

    def issue(self, code, path, line=None):
        # Never include exception messages, malformed lines, or field values.
        issue = {"code": code, "source_path": path}
        if line is not None:
            issue["record_number"] = line
        if issue not in self.issues:
            self.issues.append(issue)

    def read(self, path, required=False):
        target = self.root / path
        # Do not follow symlinks outside the historical project.
        if not target.resolve().is_relative_to(self.root):
            self.issue("source_outside_root", path)
            return None
        try:
            raw = target.read_bytes()
        except OSError:
            if required or target.exists():
                self.issue("missing_or_unreadable_source", path)
            return None
        self.files[path] = {"file_sha256": hashlib.sha256(raw).hexdigest(),
                            "bytes": len(raw)}
        return raw

    def json_file(self, path, required=False):
        raw = self.read(path, required)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (ValueError, UnicodeError):
            self.issue("invalid_json", path)
            return None

    def add(self, text, row, path, number, role, kind, evidence, declared=False):
        if not isinstance(text, str) or not normalize(text):
            self.issue("missing_or_invalid_source_text", path, number)
            return
        key = fingerprint(text)
        record = self.records.setdefault(key, {
            "report_hash": key, "ids": set(), "source_types": set(),
            "source_revisions": set(), "events": [],
        })
        identifier = row.get("id")
        if identifier is not None:
            if isinstance(identifier, str) and ID_PATTERN.fullmatch(identifier):
                record["ids"].add(identifier)
                self.ids[identifier].add(key)
            else:
                identifier = None
                self.issue("invalid_identifier", path, number)
        record["source_types"].add(kind)
        if kind == "real_report" and declared:
            record["source_revisions"].add(DATASET_REVISION)
        event = {"source_path": path, "record_number": number,
                 "record_id": identifier, "purpose": role, "evidence": evidence,
                 "declared_dataset_role": declared}
        record["events"].append(event)
        self.file_report_hashes[path].append(key)
        if path in CANONICAL:
            self.cohorts[path].append(key)
        claimed = row.get("source_sha256")
        if claimed is not None and claimed != key:
            self.issue("source_hash_mismatch", path, number)
        if evidence in {"local_output_present", "local_teacher_output_present"}:
            self.output_rows[row_fingerprint(row)].add(key)

    def jsonl(self, path):
        raw = self.read(path, path in CANONICAL or path in REQUIRED_JSONL)
        if raw is None:
            return
        try:
            lines = raw.decode("utf-8").splitlines()
        except UnicodeError:
            self.issue("invalid_utf8", path)
            return
        valid = 0
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                self.issue("invalid_jsonl_record", path, number)
                continue
            if not isinstance(row, dict):
                self.issue("unknown_record_schema", path, number)
                continue
            valid += 1
            if path in CANONICAL:
                role, kind, _ = CANONICAL[path]
                field = "source" if kind == "diagnostic_fragment" else "report"
                evidence = "reservation_record" if role in {
                    "v3_candidate_reservation", "v3_primary_selection"
                } else "declared_historical_dataset"
                self.add(row.get(field), row, path, number, role, kind, evidence, True)
            elif path.startswith("data/pilot_v1/") and (
                    path.endswith(".teacher-journal.jsonl") or path.endswith(".zh.jsonl")):
                evidence = "local_teacher_output_present" if "teacher_generation" in row else "unknown"
                if evidence == "unknown":
                    self.issue("teacher_evidence_missing", path, number)
                self.add(row.get("report"), row, path, number, "v1_teacher_generation",
                         "real_report", evidence, True)
            elif path.startswith("results/v3/real-data/") and "report" in row and (
                    "/inputs/" in path or re.search(r"/input-\d+\.jsonl$", path)):
                self.add(row["report"], row, path, number, "historical_input_copy",
                         "unresolved_report", "input_copy_present")
            elif path.startswith("results/") and ("report" in row or (
                    "source" in row and "expected_problem" in row)):
                field = "report" if "report" in row else "source"
                kind = "unresolved_report" if field == "report" else "diagnostic_fragment"
                evidence = "local_output_present" if any(k in row for k in (
                    "result", "prediction", "error")) else "unknown"
                if evidence == "unknown":
                    self.issue("output_evidence_missing", path, number)
                role = "historical_generation_or_attempt"
                if "probe" in path:
                    role = "runtime_or_model_probe"
                elif "dev" in path:
                    role = "historical_development_generation"
                elif "final" in path or "/collected/" in path:
                    role = "historical_final_generation"
                self.add(row.get(field), row, path, number, role, kind, evidence)
            else:
                self.issue("unknown_record_schema", path, number)
                if "report" in row:
                    self.add(row["report"], row, path, number, "unclassified_input",
                             "unresolved_report", "unknown")
        self.files[path]["parsed_records"] = valid
        if path in CANONICAL and valid != CANONICAL[path][2]:
            self.issue("canonical_count_mismatch", path)
        if path in CANONICAL:
            hashes = self.cohorts.get(path, [])
            # Diagnostic pairs intentionally repeat a source with good/bad targets.
            if CANONICAL[path][1] != "diagnostic_fragment" and len(set(hashes)) != len(hashes):
                self.issue("canonical_duplicate_report_hash", path)

    def known_json_sources(self):
        for path, (field, text_field, evidence, kind) in JSON_SOURCES.items():
            obj = self.json_file(path, required=True)
            if obj is None:
                continue
            rows = obj if field is None else obj.get(field) if isinstance(obj, dict) else None
            if not isinstance(rows, list):
                self.issue("unknown_record_schema", path)
                continue
            for number, row in enumerate(rows, 1):
                if not isinstance(row, dict):
                    self.issue("unknown_record_schema", path, number)
                    continue
                self.add(row.get(text_field), row, path, number, "historical_diagnostic_or_review",
                         kind, evidence)

    def code_probes(self):
        # Read literal synthetic inputs without importing/executing their scripts.
        for path, variable in (("scripts/audit_v2_guard.py", "PAIRS"),
                               ("scripts/smoke_gpu.py", "row")):
            raw = self.read(path, required=True)
            if raw is None:
                continue
            try:
                tree = ast.parse(raw)
                values = [ast.literal_eval(node.value) for node in tree.body
                          if isinstance(node, ast.Assign) and any(
                              isinstance(target, ast.Name) and target.id == variable
                              for target in node.targets)]
                value = values[0]
                if variable == "PAIRS":
                    for number, pair in enumerate(value, 1):
                        self.add(pair[2], {"id": pair[0]}, path, number,
                                 "handwritten_guard_diagnostic", "diagnostic_fragment",
                                 "declared_synthetic_probe")
                else:
                    receipt = self.json_file("results/pilot_v1/smoke.json", required=True)
                    evidence = "probe_execution_receipt_present" if isinstance(receipt, dict) and (
                        receipt.get("finite_nonzero_gradients") is True
                    ) else "declared_synthetic_probe"
                    self.add(value.get("report"), {}, path, 1, "v1_training_smoke",
                             "synthetic_probe", evidence)
            except (ValueError, SyntaxError, IndexError, TypeError, AttributeError):
                self.issue("unreadable_literal_probe", path)
        for path in ("examples/中文报告.txt", "examples/英文报告.txt"):
            raw = self.read(path, required=True)
            if raw is not None:
                try:
                    self.add(raw.decode("utf-8"), {}, path, 1, "cli_example_input",
                             "synthetic_probe", "declared_synthetic_probe")
                except UnicodeError:
                    self.issue("invalid_utf8", path)

    def review_evidence(self):
        # Record review artifacts only; never claim which person/AI saw a text.
        for base in HISTORICAL_DIRS:
            if not base.startswith("results/"):
                continue
            for target in sorted((self.root / base).rglob("*.json")):
                if "review" not in target.name or target.name == "case-review.json":
                    continue
                path = target.relative_to(self.root).as_posix()
                obj = self.json_file(path)
                if not isinstance(obj, dict) or not isinstance(obj.get("cases"), list):
                    continue
                for number, row in enumerate(obj["cases"], 1):
                    if not isinstance(row, dict):
                        self.issue("unknown_review_schema", path, number)
                        continue
                    output_hash = row.get("output_sha256")
                    keys = self.output_rows.get(output_hash, set()) if isinstance(output_hash, str) else set()
                    evidence = "sha_bound_review_record_present"
                    if not output_hash:
                        identifier = row.get("id")
                        keys = self.ids.get(identifier, set()) if isinstance(identifier, str) else set()
                        evidence = "id_linked_review_record_present"
                    if not keys:
                        self.issue("unmatched_review_record", path, number)
                    for key in keys:
                        self.records[key]["events"].append({
                            "source_path": path, "record_number": number,
                            "record_id": row.get("id") if isinstance(row.get("id"), str)
                            and ID_PATTERN.fullmatch(row["id"]) else None,
                            "purpose": "historical_review_artifact", "evidence": evidence,
                            "declared_dataset_role": False,
                        })

    def cli_receipts(self):
        path = "results/v2/final_cli_smoke.json"
        receipt = self.json_file(path, required=True)
        if isinstance(receipt, dict) and receipt.get("completed") is True:
            matches = [source for source in ("examples/中文报告.txt", "examples/英文报告.txt")
                       if self.files.get(source, {}).get("file_sha256") == receipt.get("input_sha256")]
            if not matches:
                self.issue("unmatched_cli_input_hash", path)
            for source in matches:
                for key in self.file_report_hashes[source]:
                    self.records[key]["events"].append({
                        "source_path": path, "record_number": 1, "record_id": None,
                        "purpose": "cli_smoke_execution",
                        "evidence": "sha_bound_cli_execution_receipt_present",
                        "declared_dataset_role": False,
                    })
        path = "results/mac_4bit_v1/cli_examples.json"
        receipt = self.json_file(path, required=True)
        if not isinstance(receipt, dict) or not isinstance(receipt.get("cases"), list):
            if receipt is not None:
                self.issue("unknown_cli_receipt_schema", path)
            return
        for number, row in enumerate(receipt["cases"], 1):
            if not isinstance(row, dict):
                self.issue("unknown_cli_receipt_schema", path, number)
                continue
            command = row.get("command")
            if not isinstance(command, str):
                continue
            for source in ("examples/中文报告.txt", "examples/英文报告.txt"):
                if source in command:
                    for key in self.file_report_hashes.get(source, []):
                        self.records[key]["events"].append({
                            "source_path": path, "record_number": number, "record_id": None,
                            "purpose": "cli_example_execution",
                            "evidence": "path_linked_cli_receipt_current_input_hash_unknown",
                            "declared_dataset_role": False,
                        })

    def integrity(self):
        for path in ("data/pilot_v1/manifest.json", "examples/v2/real_manifest.json",
                     "examples/v2/holdout_manifest.json", "examples/v3/real_manifest.json",
                     "examples/v3/main20_manifest.json", "examples/v3/final_candidate.lock.json",
                     "examples/v3/10-final20.json", "examples/v3/11-ablation10.json",
                     "results/v3/final/collection_manifest.json"):
            obj = self.json_file(path, required=True)
            if isinstance(obj, dict):
                self.manifests[path] = obj
                if "revision" in obj and obj["revision"] != DATASET_REVISION:
                    self.issue("unexpected_dataset_revision", path)
            elif obj is not None:
                self.issue("unknown_manifest_schema", path)
        bindings = [
            ("examples/v3/real_manifest.json", "data/v3/real/holdout.jsonl", "input_sha256"),
            ("examples/v3/main20_manifest.json", "data/v3/real/main20.jsonl", "input_sha256"),
            ("examples/v2/holdout_manifest.json", "examples/v2/holdout.jsonl", "sha256"),
        ]
        for meta, source, field in bindings:
            if meta in self.manifests and source in self.files:
                if self.manifests[meta].get(field) != self.files[source]["file_sha256"]:
                    self.issue("manifest_file_hash_mismatch", source)
        v2 = self.manifests.get("examples/v2/real_manifest.json", {})
        splits = v2.get("splits", {})
        if not isinstance(splits, dict):
            self.issue("unknown_manifest_schema", "examples/v2/real_manifest.json")
            splits = {}
        for split in ("dev", "holdout"):
            source = f"data/v2/real/{split}.jsonl"
            details = splits.get(split, {})
            if not isinstance(details, dict):
                self.issue("unknown_manifest_schema", "examples/v2/real_manifest.json")
                continue
            expected = details.get("file_sha256")
            if expected and source in self.files and self.files[source]["file_sha256"] != expected:
                self.issue("manifest_file_hash_mismatch", source)
        raw = self.read(PARQUET, required=True)
        if raw is not None and self.files[PARQUET]["file_sha256"] != PARQUET_SHA256:
            self.issue("source_parquet_hash_mismatch", PARQUET)
        original = self.cohorts.get("data/v3/real/holdout.jsonl", [])
        primary = self.cohorts.get("data/v3/real/main20.jsonl", [])
        if original and primary and primary != original[::2]:
            self.issue("v3_primary_selection_mismatch", "data/v3/real/main20.jsonl")
        for arm, (path, expected_count) in FINAL_ARMS.items():
            observed = self.file_report_hashes.get(path, [])
            expected = original[::4] if arm == "without_ledger" else primary
            if observed and (len(observed) != expected_count or observed != expected):
                self.issue("v3_final_cohort_mismatch", path)

    def build(self):
        paths = set(CANONICAL) | REQUIRED_JSONL
        for base in HISTORICAL_DIRS:
            paths.update(p.relative_to(self.root).as_posix()
                         for p in (self.root / base).rglob("*.jsonl"))
        # Explicit boundaries exclude data/results/examples p0 and later versions.
        for path in sorted(paths, key=lambda x: (x not in CANONICAL, x)):
            self.jsonl(path)
        self.known_json_sources()
        self.code_probes()
        self.review_evidence()
        self.cli_receipts()
        self.integrity()
        return self.finish()

    def finish(self):
        records = []
        used_evidence = {"local_output_present", "local_teacher_output_present",
                         "local_diagnostic_result_present", "probe_execution_receipt_present",
                         "sha_bound_cli_execution_receipt_present"}
        for key, value in sorted(self.records.items()):
            evidence = {e["evidence"] for e in value["events"]}
            kinds = value["source_types"] - {"unresolved_report"}
            if not kinds:
                kinds = {"unknown"}
                self.issue("unresolved_report_origin", value["events"][0]["source_path"],
                           value["events"][0]["record_number"])
            status = "actual_used" if evidence & used_evidence else (
                "reserved_only" if "reservation_record" in evidence else "unknown")
            if status == "unknown":
                self.issue("current_text_usage_evidence_unknown", value["events"][0]["source_path"],
                           value["events"][0]["record_number"])
            record = dict(value, ids=sorted(value["ids"]), source_types=sorted(kinds),
                          source_revisions=sorted(value["source_revisions"]) or ["unknown"],
                          usage_status=status,
                          local_output_present="local_output_present" in evidence,
                          local_teacher_output_present="local_teacher_output_present" in evidence,
                          review_evidence_present=any("review_record_present" in e for e in evidence),
                          human_or_project_ai_viewing_status="not_inferred")
            records.append(record)
        index = {r["report_hash"]: r for r in records}
        cohorts = {}
        for path, (_, kind, expected) in CANONICAL.items():
            hashes = set(self.cohorts.get(path, []))
            cohorts[path] = {"expected_records": expected, "unique_hashes": len(hashes),
                             "source_type": kind,
                             "usage_status_counts": dict(Counter(index[k]["usage_status"] for k in hashes))}
        return {
            "schema_version": 1,
            "scope": "Existing local V1, V2 and V3 files only; no new evaluation cohort created",
            "hash_algorithm": "sha256(UTF-8(re.sub(r'\\s+', ' ', text.casefold()).strip()))",
            "review_row_hash_algorithm": "sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())",
            "excluded_roots": ["data/p0", "examples/p0", "results/p0", "ops", "work", "STATUS.md", ".git", "artifacts", "cache", "logs"],
            "limitations": {
                "inventory_complete_scope": "Registered local V1--V3 files and record evidence only; not proof of complete historical, template, near-duplicate, patient, or pretraining independence",
                "outputs_seen_false_is_current_unseen_evidence": False,
                "reserved_only_meaning": "Reservation exists, but no local generation evidence was found; not a claim of unviewed or unseen text",
                "template_and_near_duplicate_audit": "pending",
                "patient_independence": "unknown",
                "diagnostic_fragment_parent_linkage": "pending",
                "parquet_rows_inspected": False,
                "foundation_model_pretraining_exposure": "unknown",
                "historical_sources_outside_local_snapshot": "unknown",
                "new_final_set_available": False,
            },
            "summary": {"unique_hashes": len(records),
                        "usage_status_counts": dict(Counter(r["usage_status"] for r in records)),
                        "issues_count": len(self.issues),
                        "inventory_complete": not self.issues},
            "v3_final_execution": {
                "arms": {arm: {"source_path": path, "expected_records": count,
                               "observed_records": len(self.file_report_hashes.get(path, [])),
                               "unique_report_hashes": len(set(self.file_report_hashes.get(path, [])))}
                         for arm, (path, count) in FINAL_ARMS.items()},
                "unique_reports_across_arms": len(set().union(*(
                    set(self.file_report_hashes.get(path, [])) for path, _ in FINAL_ARMS.values()))),
                "ablation_is_primary_subset": set(self.file_report_hashes.get(FINAL_ARMS["without_ledger"][0], []))
                    <= set(self.cohorts.get("data/v3/real/main20.jsonl", [])),
            },
            "cohorts": cohorts, "sources": dict(sorted(self.files.items())),
            "issues": self.issues, "reports": records,
        }


def build_manifest(root):
    return Inventory(root).build()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output directory already exists; refusing to overwrite")
    if not args.root.is_dir():
        parser.error("Project root does not exist")
    manifest = build_manifest(args.root)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], sort_keys=True))
    return 0 if manifest["summary"]["inventory_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
