"""Validate fixed source/target review bindings and aggregate project-AI labels.

This tool never produces review labels. It summarizes existing authored labels
after verifying the ordered packet and frozen manifest, without exporting text.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.build_exposure_manifest import fingerprint


REVIEWER = "project_AI_not_independent_clinical"
LABELS = ("observed_factual_error", "required_information_missing", "unjudgeable")
ASSESSMENTS = {"supported", "false_alarm", "mixed", "no_flag_observed_issue",
               "no_flag_no_observed_issue", "unclear"}
BINDINGS = ("id", "normalized_source_sha256", "target_sha256", "bucket")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(rows):
    result = {"cases": len(rows)}
    result.update({key: sum(row[key] for row in rows) for key in LABELS})
    result["flag_assessment"] = dict(sorted(Counter(row["flag_assessment"] for row in rows).items()))
    result["overlap_combinations"] = dict(sorted(Counter(
        "|".join(key + "=" + str(row[key]).lower() for key in LABELS)
        for row in rows).items()))
    return result


def summarize(root, manifest_path, packet_path, review_paths):
    root = Path(root)
    manifest = json.loads((root / manifest_path).read_text(encoding="utf-8"))
    if digest(root / packet_path) != manifest["packet_sha256"]:
        raise ValueError("Packet SHA does not match frozen manifest")
    packet = [json.loads(line) for line in (root / packet_path).read_text(encoding="utf-8").splitlines()
              if line.strip()]
    frozen = manifest["cases"]
    reviewed = []
    for path in review_paths:
        value = json.loads((root / path).read_text(encoding="utf-8"))
        if value.get("reviewer_type") != REVIEWER or value.get("human_reviews") != 0:
            raise ValueError("Unexpected reviewer type or human-review claim")
        reviewed.extend(value["cases"])
    if len(packet) != len(frozen) or len(reviewed) != len(frozen):
        raise ValueError("Packet, manifest and reviews must have identical counts")
    if (len({row["id"] for row in reviewed}) != len(reviewed) or
            len({row["normalized_source_sha256"] for row in reviewed}) != len(reviewed)):
        raise ValueError("Repeated review ID or normalized source hash")
    for position, (source, expected, review) in enumerate(zip(packet, frozen, reviewed), 1):
        for key in BINDINGS:
            if source[key] != expected[key] or review[key] != expected[key]:
                raise ValueError("Ordered metadata binding mismatch at position " + str(position))
        if source["flags"] != expected["flags"]:
            raise ValueError("Frozen rule flag mismatch at position " + str(position))
        if fingerprint(source["report"]) != expected["normalized_source_sha256"]:
            raise ValueError("Normalized source SHA mismatch at position " + str(position))
        if hashlib.sha256(source["reference_zh"].encode("utf-8")).hexdigest() != expected["target_sha256"]:
            raise ValueError("Target SHA mismatch at position " + str(position))
        if any(type(review[key]) is not bool for key in LABELS):
            raise ValueError("Review labels must be booleans")
        if review["flag_assessment"] not in ASSESSMENTS:
            raise ValueError("Unexpected flag assessment")
    bucket_sizes = Counter(row["bucket"] for row in frozen)
    if any(size != manifest["per_bucket"] for size in bucket_sizes.values()):
        raise ValueError("Frozen bucket size differs from predeclared count")
    for path, expected in manifest["inputs"].items():
        if digest(root / path) != expected:
            raise ValueError("Frozen source-file SHA mismatch: " + path)
    paired = list(zip(reviewed, packet))
    buckets = sorted(bucket_sizes)
    summary = {
        "schema_version": 1,
        "purpose": "Descriptive aggregation of the preselected old-training-target project-AI audit",
        "reviewer_type": REVIEWER, "human_reviews": 0,
        "inputs_sha256": {str(path): digest(root / path)
                          for path in [manifest_path, packet_path, *review_paths]},
        "frozen_upstream_inputs_sha256": manifest["inputs"],
        "integrity": {
            "manifest_records": len(frozen), "packet_records": len(packet),
            "review_records": len(reviewed), "unique_review_ids": len({r["id"] for r in reviewed}),
            "unique_normalized_source_hashes": len({r["normalized_source_sha256"] for r in reviewed}),
            "unique_target_hashes": len({r["target_sha256"] for r in reviewed}),
            "source_text_hashes_verified": len(packet), "target_text_hashes_verified": len(packet),
            "ordered_id_source_target_bucket_bindings_verified": True,
            "frozen_rule_flags_verified": True, "packet_file_sha_verified": True,
            "upstream_file_shas_verified": True, "duplicate_ids_or_sources": 0,
        },
        "overall": counts(reviewed),
        "by_preselected_bucket": {
            bucket: counts([row for row in reviewed if row["bucket"] == bucket]) for bucket in buckets},
        "by_actual_flag_presence": {
            "has_rule_flag": counts([review for review, source in paired if source["flags"]]),
            "no_rule_flag": counts([review for review, source in paired if not source["flags"]]),
        },
        "bucket_by_actual_flag_presence": {
            bucket: {
                "has_rule_flag": sum(review["bucket"] == bucket and bool(source["flags"])
                                     for review, source in paired),
                "no_rule_flag": sum(review["bucket"] == bucket and not source["flags"]
                                    for review, source in paired),
            } for bucket in buckets},
        "label_conventions": {
            "overlap": "All three boolean outcomes may overlap; their counts must not be added as disjoint case totals.",
            "observed_factual_error": "An authored project-AI judgment of explicit source inconsistency or unsupported case-specific assertion; not an image-truth or clinical-accuracy judgment.",
            "required_information_missing": "Includes loss of explicit source qualifiers, relations, anatomic specificity and technical acquisition information. The field name does not mean independently clinician-adjudicated clinical necessity or harm.",
            "unjudgeable": "At least one aspect remains unresolved, including source ambiguity, redaction, internal inconsistency or terminology; other aspects may still support definite findings.",
            "flag_assessment": "Assesses existing rule flags separately from other observed target problems. A false alarm can coexist with an unrelated observed target error; no-flag observed issue may be an omission without a definite factual error.",
        },
        "limitations": [
            "This is project-AI-assisted review, not independent clinical review; human reviews completed: zero.",
            "The fixed stratified eighteen-case exploratory sample is not a population estimate or unbiased accuracy estimate.",
            "The random_remaining bucket was sampled only after removing the other twelve selected cases; it is not an unbiased sample of all training data.",
            "Counts describe these selected source/target pairs, not filter sensitivity, specificity, clinical correctness or unseen model performance.",
            "The absence of an observed issue is not proof of correctness. Medical terminology and source ambiguity may still require qualified human adjudication.",
            "No review labels, training-screening rules, historical data or final evaluation selections were changed by this summarizer.",
        ],
    }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.root / args.out
    if output.exists():
        parser.error("Refusing to overwrite an existing review summary")
    summary = summarize(args.root, args.manifest, args.packet, args.reviews)
    summary["summarizer_sha256"] = digest(Path(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
