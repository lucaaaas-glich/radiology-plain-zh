"""Meaningful boundary and provenance tests for conservative source grouping."""
import itertools
import json
from pathlib import Path

from scripts.group_p0_sources import (
    build_group_map, find_fragment_parents, near_pairs, numeric_template,
    shingles, split_conflicts,
)


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/p0_grouping.json").read_text())


def test_exact_prefix_join_matches_exhaustive_oracle_including_threshold_boundary():
    # Includes empty sets, all overlap patterns, unequal sizes and ties in token
    # frequency. This catches candidate pruning errors, not only final scoring.
    universe = range(7)
    sets = {str(index): set(values) for index, values in enumerate(
        values for length in range(8) for values in itertools.combinations(universe, length))}
    actual = {frozenset((a, b)) for a, b, _, _ in near_pairs(sets)}
    expected = {frozenset((a, b)) for a, b in itertools.combinations(sets, 2)
                if sets[a] and sets[b]
                and 5 * len(sets[a] & sets[b]) >= 4 * len(sets[a] | sets[b])}
    assert actual == expected
    boundary = {"short": {1, 2, 3, 4}, "long": {1, 2, 3, 4, 5}, "below": {1, 2, 3, 6}}
    matches = {frozenset((a, b)): (i, u) for a, b, i, u in near_pairs(boundary)}
    assert matches == {frozenset(("short", "long")): (4, 5)}


def test_transitive_components_keep_non_directly_matching_reports_together():
    sets = {"a": set(range(8)), "b": set(range(9)), "c": set(range(1, 9))}
    edges = [{"type": "near", "left": a, "right": b} for a, b, _, _ in near_pairs(sets)]
    assert len(edges) == 2  # a-c Jaccard is 7/9, below the 0.8 threshold.
    assert build_group_map(sets, edges) == {"a": "a", "b": "a", "c": "a"}


def test_transitive_components_and_representatives_are_order_independent():
    edges = [{"type": "near", "left": "b", "right": "c"},
             {"type": "template", "left": "a", "right": "b"}]
    expected = {"a": "a", "b": "a", "c": "a", "d": "d"}
    assert build_group_map("abcd", edges) == expected
    assert build_group_map("dcba", list(reversed(edges))) == expected


def test_short_templates_preserve_negation_anatomy_units_and_punctuation():
    settings = CONFIG["short_numeric_template"]
    template = numeric_template("No left mass of 1.5 cm.", settings)
    assert template == numeric_template(" NO left mass of 27.4 cm. ", settings)
    assert template != numeric_template("Left mass of 1.5 cm.", settings)
    assert template != numeric_template("No right mass of 1.5 cm.", settings)
    assert template != numeric_template("No left mass of 1.5 mm.", settings)
    assert numeric_template("No left mass.", settings) is None
    assert numeric_template("word " * 81 + "1", settings) is None
    assert numeric_template("a" * 701 + " 1", settings) is None


def test_legacy_shingles_match_punctuation_case_and_short_rules():
    assert shingles("ONE, two three FOUR five six.") == {
        ("one", "two", "three", "four", "five"), ("two", "three", "four", "five", "six")}
    assert not shingles("one two three four")
    assert not list(near_pairs({"short1": set(), "short2": set()}))


def test_fragment_containment_records_all_candidates_without_merging_parents():
    texts = {"fragment": "No pleural effusion.",
             "patient_a": "No pleural effusion. A left lung opacity is present.",
             "patient_b": "A right lung mass is present. No pleural effusion.",
             "boundary": "pseudoNo pleural effusion."}
    links = find_fragment_parents(texts, {"fragment"}, {"patient_a", "patient_b", "boundary"})
    assert links == {"fragment": ["patient_a", "patient_b"]}
    possible = [{"type": "possible_parent", "left": "fragment", "right": key}
                for key in links["fragment"]]
    groups = build_group_map(texts, possible)
    assert groups["patient_a"] != groups["patient_b"]
    assert groups["fragment"] not in {groups["patient_a"], groups["patient_b"]}


def test_cross_split_counts_include_transitive_conflict_without_direct_edge():
    edges = [{"type": "near", "left": "train", "right": "bridge"},
             {"type": "template", "left": "bridge", "right": "dev"}]
    groups = build_group_map(["train", "bridge", "dev", "test"], edges)
    counts = split_conflicts(groups, {"train": {"train"}, "validation": {"dev"}, "test": {"test"}}, edges)
    assert counts["train__validation"]["shared_components"] == 1
    assert counts["train__validation"]["cross_split_report_pairs_in_shared_components"] == 1
    assert counts["train__validation"]["direct_edges_by_type"] == {}
    assert counts["train__test"]["shared_components"] == 0
