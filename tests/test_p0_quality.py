"""Meaningful P0 risk-screen invariants; no model or medical truth assertions."""
from pathlib import Path
import json

import pytest

from radplain.quality_p0 import (risk_details, screen_row, summarize_rows,
                                quantity_counter, calibration_report)
from scripts.screen_p0_training import validate_calibration


def codes(source, target):
    return {i["code"] for i in risk_details(source, target)}


def test_valid_unit_conversion_reordering_and_numbered_list_not_flagged():
    assert not codes("1. Nodule 2 cm. 2. Previously 8 mm.", "以前8毫米的结节，目前20毫米。")
    assert not codes("Mild bibasal atelectasis.", "双肺底部有少量肺组织没有充分展开。")
    assert not codes("COVID-19. A 4 mm nodule.", "肺部4毫米结节，COVID-19。")


def test_quantity_changes_multiplicity_shared_unit_and_explicit_bounds():
    assert "quantity_mismatch" in codes("Nodule 4 mm.", "结节5毫米。")
    assert "quantity_mismatch" in codes("Two nodules: 4 mm, 4 mm.", "结节4毫米。")
    assert "quantity_mismatch" in codes("Follow-up in 4–6 weeks.", "6周后复查。")
    assert quantity_counter("12 x 6 mm") == quantity_counter("12毫米×6毫米")
    assert not codes("Less than 1 cm.", "小于10毫米。")
    assert "explicit_size_bound_mismatch" in codes("Less than 1 cm.", "10毫米。")


def test_date_reformat_reorder_and_ambiguity():
    assert not codes("Compared with 2022-04-13 and 2021-03-17.", "与2021年3月17日及2022年4月13日比较。")
    assert "ambiguous_date_reinterpreted" in codes("Compared with 03/07/2022.", "与2022年3月7日比较。")
    assert not codes("Surgery in 2015.", "2015年手术。")
    assert "calendar_year_mismatch" in codes("Surgery in 2015.", "2016年手术。")


def test_explicit_markers_and_known_association_blindspot():
    assert "explicit_negation_marker_missing" in codes("No pneumothorax.", "存在气胸。")
    assert not codes("Malignancy cannot be excluded.", "不能排除恶性病变。")
    assert "uncertainty_possible_missing" in codes("Possible infection.", "存在感染。")
    assert "uncertainty_possible_missing" in codes("Possibly infection.", "存在感染。")
    assert "exclusive_laterality_mismatch" in codes("Left effusion.", "右侧积液。")
    assert "prior_measurement_mismatch" in codes("4 mm, previously 2 mm.", "以前4毫米，现在2毫米。")
    # Deliberate evidence that an empty flag list cannot certify relationships.
    assert not codes("Liver lesion 4 mm; renal lesion 2 mm.", "肾脏病灶4毫米；肝脏病灶2毫米。")


def test_preserve_format_rejected_rows_and_aggregate_shift_without_text():
    base = dict(source="synthetic", report="A 3 mm nodule.", reference_zh="3毫米结节。")
    kept, _ = screen_row(dict(base, id="kept", teacher_accepted=True))
    risk, _ = screen_row(dict(base, id="risk", teacher_accepted=True, reference_zh="4毫米结节。"))
    rejected, _ = screen_row(dict(base, id="rejected", teacher_accepted=False))
    assert rejected["screen_status"] == "teacher_format_rejected"
    assert kept["screen_status"] == "no_known_rule_flag"
    assert risk["screen_status"] == "risk_flagged"
    assert len(kept["source_sha256"]) == 64 and "report" not in kept and "reference_zh" not in kept
    summary = summarize_rows([kept, risk, rejected])
    assert summary["counts"] == {"all_training_rows": 3, "teacher_format_accepted": 2,
        "teacher_format_rejected": 1, "accepted_with_risk_flag": 1,
        "accepted_no_known_rule_flag": 1, "new_human_reviews": 0}
    assert summary["before_after_distribution_shift"]["quantity_presence"]["true"]["after_count"] == 1
    assert "A 3 mm nodule" not in json.dumps(summary)
    with pytest.raises(ValueError, match="explicit boolean"):
        screen_row(dict(base, id="invalid", teacher_accepted="true"))


def test_calibration_hash_and_family_split_are_frozen_and_development_only():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/p0_quality.json").read_text())
    pairs = validate_calibration(root / config["calibration_path"], config)
    summary, outcomes = calibration_report(pairs, "development")
    assert len(pairs) == 60 and summary["pair_count"] == 40 and len(outcomes) == 80
    assert summary["counts"]["control_candidates"] == 40
    assert summary["counts"]["known_perturbations"] == 40
    assert not any(p["human_reviewed"] for p in pairs)
    broken = dict(config, calibration_sha256="0" * 64)
    with pytest.raises(ValueError, match="Calibration changed"):
        validate_calibration(root / config["calibration_path"], broken)
