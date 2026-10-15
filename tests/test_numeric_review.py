from radplain.fidelity import numeric_review


def test_flags_observed_wrong_derived_growth_without_changing_text():
    out = numeric_review("8 mm, previously 5 mm. CT in 3 months.",
                         "从5毫米变成8毫米，增大了1毫米，三个月后复查。")
    assert out["extra_output_quantities"] == ["1毫米"]
    assert not out["unmatched_source_quantities"]


def test_equivalent_length_and_calendar_units():
    out = numeric_review("30 mm; 6 months; 2 weeks.", "3厘米；半年；14天。")
    assert not out["warnings"]


def test_flags_missing_old_measurement_and_novel_decimal():
    out = numeric_review("约7mm，较前6mm略增大，6个月后复查。", "7毫米，增大零点五毫米，半年后复查。")
    assert out["extra_output_quantities"] == ["零点五毫米"]
    assert out["unmatched_source_quantities"] == ["6mm"]


def test_does_not_claim_to_detect_swapped_relations():
    out = numeric_review("8 mm, previously 5 mm.", "现在5毫米，以前8毫米。")
    assert not out["warnings"]  # Known limitation: matching values do not prove matching meaning.
    assert "not_semantic_or_medical" in out["scope"]


def test_malformed_free_text_does_not_crash_checker():
    assert not numeric_review("点毫米", "零点点毫米")["warnings"]
