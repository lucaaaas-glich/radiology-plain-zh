import pytest
from radplain.common import assert_disjoint, chinese_ratio, fingerprint
from radplain.data import clean_rows


def test_report_dedup_ignores_whitespace_and_case():
    assert fingerprint("No  pleural\nEFFUSION") == fingerprint("no pleural effusion")


def test_cross_split_leakage_is_rejected_even_with_different_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        assert_disjoint({"train": [{"id": "a", "report": "No pleural effusion"}],
                         "test": [{"id": "b", "report": "no  pleural effusion"}]})


def test_no_empty_or_duplicate_reference_pairs():
    rows = [{"radiology_report": "The heart is mildly enlarged.", "layman_report": "The heart is a little larger than usual."},
            {"radiology_report": "THE HEART IS MILDLY ENLARGED.", "layman_report": "Different wording but the same source."},
            {"radiology_report": "There is a right pleural effusion.", "layman_report": None}]
    clean, counts = clean_rows(rows)
    assert len(clean) == 1
    assert counts["duplicate_report"] == 1
    assert counts["empty_or_outside_character_range"] == 1


def test_chinese_detection_does_not_count_numbers_as_chinese():
    assert chinese_ratio("右肺有一个6毫米的小结节。") > 0.8
    assert chinese_ratio("123456 mm no abnormality") == 0
