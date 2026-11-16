"""Synthetic 40x3 review contracts; never read actual predictions or mappings."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.summarize_p1_comparison import main as aggregate


BASE = Path("data/p1/comparison-20261117")
LOCAL = BASE / "blind-review"
PUBLIC = Path("results/p1/comparison-20261117")
PARTS = ((1, 14), (15, 27), (28, 40))
PRIVATE_NOTE = "SYNTHETIC_PRIVATE_NOTE_NEVER_EXPORT_原文引用"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def synthetic_cohort():
    packets, reviews, mapping, source_facts = [], [], [], []
    for position in range(1, 41):
        # One longer synthetic report distinguishes report-macro from fact-micro.
        count = 4 if position == 2 else 1
        report = "SYNTHETIC ONLY " + str(position) + ". " + " ".join(
            "Fact " + str(i) + "." for i in range(1, count + 1))
        key, source_sha = text_digest("synthetic-id-" + str(position)), text_digest(report)
        facts = []
        for i in range(1, count + 1):
            quote = "Fact " + str(i) + "."
            start = report.index(quote)
            facts.append({"fact_id": f"F{i:02}", "source_evidence": quote,
                          "start": start, "end": start + len(quote),
                          "meaning_zh": "合成事实" + str(i), "required": True})
        sf = {"position": position, "id": key, "source_sha256": source_sha,
              "reviewer": "synthetic_source_reviewer", "human_review": False,
              "candidate_outputs_seen": False, "required_facts": facts, "source_notes": []}
        source_facts.append(sf)
        packet = {"position": position, "id": key, "report": report,
                  "source_sha256": source_sha, "source_facts": sf, "candidates": []}
        review = {"position": position, "id": key, "source_sha256": source_sha,
                  "human_review": False, "candidate_arm_names_seen": False,
                  "reviewer": "synthetic_blind_reviewer", "candidates": []}
        for label, arm in zip(("A", "B", "C"), ("base", "S0", "S1")):
            prediction = f"合成案例{position}候选{label}，所有合成事实均保留。"
            prediction_sha = text_digest(prediction)
            packet["candidates"].append({"candidate": label, "prediction": prediction,
                "prediction_sha256": prediction_sha, "generation_completed": True,
                "error_code": None, "eos_completed": True})
            review["candidates"].append({"candidate": label, "prediction_sha256": prediction_sha,
                "source_error": False, "incomplete": False, "unjudgeable": False,
                "precision_issue": False, "chinese_complete": True, "retained_source_text": False,
                "refusal": False, "readability": "fluent",
                "fact_assessments": [{"fact_id": f["fact_id"], "status": "faithful",
                                      "note_zh": PRIVATE_NOTE} for f in facts],
                "issues": [], "notes_zh": PRIVATE_NOTE})
            mapping.append({"position": position, "id": key, "candidate": label,
                            "arm": arm, "prediction_sha256": prediction_sha})
        packets.append(packet)
        reviews.append(review)
    return {"packets": packets, "reviews": reviews, "mapping": mapping, "facts": source_facts}


def set_prediction(cohort, position, label, prediction):
    index = "ABC".index(label)
    sha = text_digest(prediction)
    cohort["packets"][position - 1]["candidates"][index].update(
        prediction=prediction, prediction_sha256=sha)
    cohort["reviews"][position - 1]["candidates"][index]["prediction_sha256"] = sha
    for entry in cohort["mapping"]:
        if entry["position"] == position and entry["candidate"] == label:
            entry["prediction_sha256"] = sha


def fail_generation(cohort, position, label, error_code, *, empty=False, eos=False):
    index = "ABC".index(label)
    if empty:
        set_prediction(cohort, position, label, "")
    prediction = cohort["packets"][position - 1]["candidates"][index]
    prediction.update(generation_completed=False, error_code=error_code, eos_completed=eos)
    review = cohort["reviews"][position - 1]["candidates"][index]
    review.update(incomplete=True, chinese_complete=False)
    if empty:
        for item in review["fact_assessments"]:
            item["status"] = "missing"
        review["readability"] = "hard_to_read"
        review["issues"] = [{"type": "missing_fact", "note_zh": PRIVATE_NOTE},
                            {"type": "malformed public issue: " + PRIVATE_NOTE,
                             "notes_zh": PRIVATE_NOTE}]


def freeze_fixture(root, cohort):
    source_paths, packet_paths, review_paths = [], [], []
    for lo, hi in PARTS:
        suffix = f"{lo:02}_{hi:02}"
        source_path = BASE / "source-review" / f"final40_{suffix}.json"
        packet_path = LOCAL / f"packet_{suffix}.json"
        review_path = LOCAL / f"review_{suffix}.json"
        write(root / source_path, cohort["facts"][lo - 1:hi])
        write(root / packet_path, cohort["packets"][lo - 1:hi])
        source_paths.append(source_path)
        packet_paths.append(packet_path)
        review_paths.append(review_path)
    fact_lock = Path("examples/p1/final_source_facts_20261117.lock.json")
    input_lock = Path("examples/p1/final40_reviewed_20261117.lock.json")
    instructions = BASE / "blind_review_instructions.txt"
    write(root / fact_lock, {"source_review_files_sha256": {
        str(p): digest(root / p) for p in source_paths}})
    write(root / input_lock, {"synthetic_fixture": True})
    (root / instructions).write_text("Synthetic-only fixture instructions. No actual cases.")
    map_path = LOCAL / "arm_mapping.private.json"
    seed_path = LOCAL / "randomization_seed.private.json"
    write(root / map_path, cohort["mapping"])
    write(root / seed_path, {"seed": 1, "synthetic_fixture": True})
    bound_paths = [*packet_paths, map_path, seed_path, fact_lock, input_lock, instructions]
    write(root / PUBLIC / "blind_packet_receipt.json", {"files_sha256": {
        str(p): digest(root / p) for p in bound_paths}})
    # Reviews are authored after the packet receipt, not included in that receipt.
    for (lo, hi), path in zip(PARTS, review_paths):
        write(root / path, cohort["reviews"][lo - 1:hi])


def run_aggregate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    aggregate()
    return json.loads((tmp_path / PUBLIC / "final_ai_comparison.json").read_text())


def test_failures_stay_in_40_report_denominators_and_private_text_stays_private(tmp_path, monkeypatch):
    cohort = synthetic_cohort()
    fail_generation(cohort, 1, "B", "generation_limit_without_eos")
    fail_generation(cohort, 2, "B", "empty_output", empty=True, eos=True)
    fail_generation(cohort, 3, "C", "input_overlength_no_truncation", empty=True)
    freeze_fixture(tmp_path, cohort)
    result = run_aggregate(tmp_path, monkeypatch)
    assert len(result["anonymous_cases"]) == 120
    assert all(a["reports"] == 40 for a in result["arms"].values())
    assert result["arms"]["base"]["complete_faithful_chinese"] == 40
    s0 = result["arms"]["S0"]
    assert s0["complete_faithful_chinese"] == 38
    assert s0["primary_status"]["incomplete"] == 2
    assert s0["generation_failure_codes"] == {"generation_limit_without_eos": 1, "empty_output": 1}
    assert s0["eos_completed"] == 39
    assert s0["facts"]["required"] == 43 and s0["facts"]["covered"] == 39
    assert s0["facts"]["coverage_micro_fraction"] == pytest.approx(39 / 43)
    assert s0["facts"]["coverage_macro_fraction"] == pytest.approx(39 / 40)
    assert result["arms"]["S1"]["complete_faithful_chinese"] == 39
    assert result["paired_comparisons"]["base_to_S0"]["complete_faithful_chinese"]["difference"] == pytest.approx(-2 / 40)
    public = json.dumps(result, ensure_ascii=False)
    assert PRIVATE_NOTE not in public
    assert "SYNTHETIC ONLY" not in public
    assert "合成案例" not in public
    assert all("notes_zh" not in row and "prediction" not in row and "report" not in row
               for row in result["anonymous_cases"])
    assert all(len(row["private_notes_sha256"]) == 64 for row in result["anonymous_cases"])


def test_identical_text_with_inconsistent_fact_labels_is_rejected(tmp_path, monkeypatch):
    cohort = synthetic_cohort()
    same = cohort["packets"][0]["candidates"][0]["prediction"]
    set_prediction(cohort, 1, "B", same)
    changed = cohort["reviews"][0]["candidates"][1]
    changed["fact_assessments"][0]["status"] = "less_precise"
    changed["precision_issue"] = True
    freeze_fixture(tmp_path, cohort)
    with pytest.raises(ValueError, match="Identical predictions have inconsistent reviews"):
        run_aggregate(tmp_path, monkeypatch)


def test_identical_text_can_have_different_terminal_eos_without_review_conflict(tmp_path, monkeypatch):
    cohort = synthetic_cohort()
    same = cohort["packets"][0]["candidates"][0]["prediction"]
    set_prediction(cohort, 1, "B", same)
    fail_generation(cohort, 1, "B", "generation_limit_without_eos")
    freeze_fixture(tmp_path, cohort)
    result = run_aggregate(tmp_path, monkeypatch)
    assert result["arms"]["base"]["complete_faithful_chinese"] == 40
    assert result["arms"]["S0"]["complete_faithful_chinese"] == 39
    assert result["arms"]["S0"]["primary_status"]["incomplete"] == 1
    assert result["arms"]["S0"]["eos_completed"] == 39
    assert result["arms"]["base"]["facts"] == result["arms"]["S0"]["facts"]


@pytest.mark.parametrize("path", [LOCAL / "packet_28_40.json", LOCAL / "arm_mapping.private.json",
                                  Path("examples/p1/final_source_facts_20261117.lock.json")])
def test_frozen_packet_mapping_or_lock_tampering_is_rejected(tmp_path, monkeypatch, path):
    freeze_fixture(tmp_path, synthetic_cohort())
    # Even whitespace changes must violate the exact sealed artifact hash.
    p = tmp_path / path
    p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError, match="Frozen blind packet or mapping changed"):
        run_aggregate(tmp_path, monkeypatch)


def test_prediction_hash_is_recomputed_not_merely_copied_from_packet(tmp_path, monkeypatch):
    cohort = synthetic_cohort()
    freeze_fixture(tmp_path, cohort)
    packet_path = tmp_path / LOCAL / "packet_01_14.json"
    packets = json.loads(packet_path.read_text())
    packets[0]["candidates"][0]["prediction"] += "合成变更"
    write(packet_path, packets)
    # Simulate a valid transfer digest with a stale per-prediction hash claim.
    receipt_path = tmp_path / PUBLIC / "blind_packet_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["files_sha256"][str(LOCAL / "packet_01_14.json")] = digest(packet_path)
    write(receipt_path, receipt)
    with pytest.raises(AssertionError):
        run_aggregate(tmp_path, monkeypatch)
