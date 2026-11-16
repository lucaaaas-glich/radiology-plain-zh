"""Old-development-derived cases only; no new frozen validation is consumed."""
import pytest

from radplain.quality_p1 import screen_pair, screen_row, date_mentions


@pytest.mark.parametrize("source,target", [
    ("No changes since March 2009.", "与2009年3月相比没有变化。"),
    ("Compared with the July study.", "与7月份的检查比较。"),
    ("Last CT dated July 2011.", "上次CT在2011年7月。"),
    ("Surgery in 2015.", "2015年手术。"),
    ("Examination on 2022-04-13.", "检查在2022年4月13日。"),
    ("Compared with July 15, 2011.", "与2011年7月15日比较。"),
    ("Compared with 15 July 2011.", "与2011年7月15日比较。"),
    ("Compared with March 2009 and July 2011.", "与2011年7月和2009年3月比较。"),
    ("Follow-up in 6 months.", "6个月后复查。"),
    ("A 2 cm lesion.", "病灶20毫米。"),
    ("A lesion may represent infection.", "病灶可能是感染。"),
])
def test_correct_development_dates_units_and_modal_may(source, target):
    assert screen_pair(source, target)["status"] == "no_supported_issue_found"


def test_ambiguous_date_selection_is_review_but_impossible_month_is_clear():
    assert screen_pair("Compared with 1 12 2020.", "与2020年1月12日比较。")["status"] == "review_needed"
    assert screen_pair("Compared with 1 12 2020.", "与2020年12月1日比较。")["status"] == "review_needed"
    result = screen_pair("Compared with 1 12 2020.", "与2020年11月12日比较。")
    assert result["status"] == "clear_issue" and "ambiguous_date_incompatible" in result["flags"]
    assert screen_pair("Compared with 1 12 2020.", "与1 12 2020比较。")["status"] == "no_supported_issue_found"


def test_multiple_dates_and_date_specificity_do_not_claim_unique_alignment():
    assert screen_pair("Compared with March 2009.", "与2009年3月5日比较。")["status"] == "review_needed"
    assert screen_pair("Compared with [date].", "与2020年1月1日比较。")["status"] == "review_needed"
    assert screen_pair("Compared with March 2009 and June 2010.", "与2009年4月和2010年6月比较。")["status"] == "review_needed"
    assert screen_pair("Compared with March 2009.", "与2009年4月比较。")["status"] == "clear_issue"
    assert not date_mentions("Changes may represent infection; follow-up in 7 months.")
    assert screen_pair("From March to July 2011.", "从2011年3月至7月。")["status"] != "clear_issue"
    assert screen_pair("Compared with 1 12 2020.", "与1 12 2020和2022年3月1日比较。")["status"] == "review_needed"


def test_same_terms_with_different_objects_or_multiple_device_mentions_not_clear():
    result = screen_pair("Consolidation decreased. A right nodule persists.", "右肺结节减少，肺部另有阴影。")
    assert result["status"] != "clear_issue"
    result = screen_pair("Endotracheal tube tip above carina. Endotracheal tube repositioned.",
                         "气管插管尖端在隆突下方。气管插管已调整。")
    assert result["status"] == "review_needed"


def test_quantity_mismatch_requires_single_same_dimension_for_clear_label():
    assert screen_pair("Lesion 4 mm.", "病灶5毫米。")["status"] == "clear_issue"
    assert screen_pair("Liver lesion 4 mm; renal lesion 2 mm.", "肝脏病灶5毫米；肾脏病灶2毫米。")["status"] == "review_needed"
    assert screen_pair("Lesion 4 mm.", "病灶。")["status"] == "review_needed"


def test_substitution_unique_finding_vs_mixed_findings():
    assert screen_pair("Bilateral consolidations decreased.", "双肺结节减少。")["status"] == "clear_issue"
    assert screen_pair("No consolidation.", "没有肺炎。")["status"] == "clear_issue"
    assert screen_pair("No consolidation. A nodule persists.", "没有实变，仍有结节。")["status"] == "no_supported_issue_found"
    assert screen_pair("No consolidation. A nodule persists.", "仍有结节。")["status"] == "review_needed"
    result = screen_pair("Consolidation and effusion.", "肺炎和积液。")
    assert result["status"] == "review_needed"
    assert screen_pair("Bilateral consolidations decreased.", "双侧实变减少。")["status"] == "no_supported_issue_found"


def test_device_chambers_are_not_device_count_and_mixed_context_is_downgraded():
    assert screen_pair("A bicameral pacemaker.", "两个心脏起搏器。")["status"] == "clear_issue"
    assert screen_pair("A bicameral pacemaker.", "两个心脏起搏器（双心起搏器）。")["status"] == "clear_issue"
    assert screen_pair("A bicameral pacemaker.", "双腔心脏起搏器。")["status"] == "no_supported_issue_found"
    assert "pacemaker_chambers_as_device_count" not in screen_pair("Two pacemakers, one dual-chamber.", "两个起搏器，其中一个双腔。")["flags"]


def test_device_endpoint_and_relation_have_distinct_certainty():
    missing = screen_pair("Nasogastric tube with its distal end below the diaphragm.", "胃管位于膈肌下方。")
    assert missing["status"] == "review_needed" and "nasogastric_tube_endpoint_not_matched" in missing["flags"]
    assert screen_pair("Nasogastric tube tip below diaphragm.", "胃管尖端位于膈肌上方。")["status"] == "clear_issue"
    assert screen_pair("Endotracheal tube tip 3.7 cm above carina.", "气管插管尖端在隆突上方3.7厘米。")["status"] == "no_supported_issue_found"


def test_device_normality_requires_unsupported_unique_device_listing():
    assert screen_pair("Ventriculoperitoneal shunt catheter.", "脑室腹腔分流管位置正常。")["status"] == "clear_issue"
    assert screen_pair("Ventriculoperitoneal shunt catheter in normal position.", "脑室腹腔分流管位置正常。")["status"] == "no_supported_issue_found"
    assert screen_pair("Ventriculoperitoneal shunt catheter. Heart size normal.", "脑室腹腔分流管。心脏大小正常。")["status"] == "no_supported_issue_found"


@pytest.mark.parametrize("source,target,expected", [
    ("Radiological resolution of pneumonia.", "影像显示肺炎有所改善。", "review_needed"),
    ("Radiological resolution of pneumonia.", "影像显示肺炎已消退。", "no_supported_issue_found"),
    ("Radiological resolution of pneumonia.", "现在的检查中看不到了。", "no_supported_issue_found"),
    ("Partial resolution of pneumonia.", "肺炎有所改善。", "no_supported_issue_found"),
    ("To assess resolution of pneumonia.", "评估肺炎是否消退。", "no_supported_issue_found"),
    ("Pneumonia has not resolved.", "肺炎仍存在。", "no_supported_issue_found"),
    ("High resolution CT.", "高分辨率CT。", "no_supported_issue_found"),
])
def test_resolved_is_not_improved_but_partial_or_planned_resolution_not_overcalled(source, target, expected):
    assert screen_pair(source, target)["status"] == expected


def test_evidence_offsets_exact_and_unreviewed_row_metadata():
    source, target = "A bicameral pacemaker.", "两个心脏起搏器。"
    result = screen_pair(source, target)
    for issue in result["issues"]:
        for side, text in (("source", source), ("target", target)):
            assert issue[side + "_evidence"]
            for ev in issue[side + "_evidence"]:
                assert text[ev["start"]:ev["end"]] == ev["text"]
    row = screen_row({"id": "test", "source": "synthetic", "report": source,
                      "reference_zh": target, "teacher_accepted": True})
    assert row["id"] == "test" and row["teacher_accepted"] is True and row["human_reviewed"] is False
    assert len(row["source_sha256"]) == len(row["target_sha256"]) == 64
