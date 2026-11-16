"""CPU-only, interpretable training-target risk screening for the P0 experiment.

An empty flag list means only ``no_known_rule_flag``. It is neither factual
approval nor human/medical annotation. The old V1/V2/V3 code is imported read-only
for selected lexical helpers; the old whole-report guard is deliberately unused.
"""
from collections import Counter
from decimal import Decimal
import hashlib
import re

from .fidelity import _QUANTITY, quantities
from .guard_v2 import FEATURES, NEG, negation_text, prior_roles
from .quantity_v2 import calendar_years, dates, normalized_quantities_text, size_bounds


VERSION = "p0-risk-screen-v1"
UNCERTAINTY = {name: (source, target) for name, source, target in FEATURES
               if name in {"indeterminate", "cannot_exclude", "possible"}}
# Development pair 022 exposed the adverb form absent from the inherited regex.
# Keep this additive normalization here rather than changing the old V2 guard.
UNCERTAINTY["possible"] = (UNCERTAINTY["possible"][0] + r"|\bpossibly\b",
                           UNCERTAINTY["possible"][1])
TEMPORAL = re.compile(r"\b(?:previously|previous|prior|since|follow[- ]?up|interval)\b|"
                      r"以前|前次|既往|此前|复查|随访|较前|上次", re.I)
SIDE = {"left": r"\bleft(?:-sided)?\b|左", "right": r"\bright(?:-sided)?\b|右"}
BILATERAL = re.compile(r"\bbilateral\b|\bboth\b|\bbibasa[lr]\b|"
                       r"双(?:侧|肺|肾)|两(?:侧|肺|肾)|左右", re.I)
RANGE_OR_DIMENSION = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:[-–—~至到]|[x×])\s*"
    r"(\d+(?:\.\d+)?)\s*(mm|cm|毫米|厘米|weeks?|months?|years?|days?|周|个月|月|年|天)(?![a-z])",
    re.I)

LIMITATIONS = [
    "Flags are lexical risk indicators; no_known_rule_flag is not factual or clinical approval.",
    "New human reviews = 0. Calibration labels are AI-authored known synthetic edits, not medical gold.",
    "English/Chinese lexical coverage is incomplete; source errors and translation errors are not distinguished.",
    "Quantities cover explicit mm/cm and day/week/month/year units; equivalent units and reordered mentions are allowed.",
    "Bare numbers, list/series/image IDs, percentages, volumes, most bounds/ranges and implicit units may be missed.",
    "Quantity counters do not bind values to organs; repeated explanation of a value may cause a false flag.",
    "Date handling reuses narrow V2 formats; ambiguous dates must remain verbatim, and named-month dates may be missed.",
    "Negation and uncertainty checks mostly detect marker loss, not finding-level scope or diagnostic correctness.",
    "Laterality checks only compare exclusive left/right markers; mixed bilateral assignments are not verified.",
    "Unsupported causes, diagnostic subtypes, anatomy translations, omissions and advice can remain unflagged.",
    "The 60 small synthetic pairs are a diagnostic stress set, not prevalence or generalization estimates.",
    "Filtering may shift source/length/feature distributions; aggregate shifts are reported, not corrected here.",
]


def sha256_text(text):
    """Exact UTF-8 hash, unlike the case/whitespace-normalized historical ID."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def quantity_counter(text):
    """Normalize supported units, preserve multiplicity, ignore bare numbering."""
    value = normalized_quantities_text(text)
    # Carry an explicit shared unit to both endpoints. This still does not
    # understand the geometric/clinical relationship between these values.
    value = RANGE_OR_DIMENSION.sub(lambda m: f"{m[1]} {m[3]} {m[2]} {m[3]}", value)
    result = Counter()
    for match in _QUANTITY.finditer(value):
        result.update(quantities(match.group()).keys())
    return result


def bound_counter(text):
    return Counter((operator, Decimal(number) * (10 if unit == "cm" else 1))
                   for operator, number, unit in size_bounds(text))


def side_set(text):
    if BILATERAL.search(text):
        return {"left", "right"}
    return {side for side, pattern in SIDE.items() if re.search(pattern, text, re.I)}


def has_uncertainty(text):
    return any(re.search(source + "|" + target, text, re.I)
               for source, target in UNCERTAINTY.values())


def features(report, source):
    return {"source": source,
            "length_bin": "short" if len(report) < 200 else "medium" if len(report) < 600 else "long",
            "quantity_presence": bool(quantity_counter(report)),
            "negation_presence": bool(NEG.search(negation_text(report))),
            "uncertainty_presence": has_uncertainty(report),
            "temporal_presence": bool(TEMPORAL.search(report) or dates(report) or calendar_years(report))}


def risk_details(source, target):
    """Return explainable details. Store these only in ignored local output."""
    issues = []

    def add(code, detail):
        issues.append({"code": code, "detail": detail})

    if not target.strip():
        add("empty_target", "Target is empty.")
    before, after = quantity_counter(source), quantity_counter(target)
    if before != after:
        def printable(counter):
            return [{"kind": kind, "normalized_value": str(value), "count": count}
                    for (kind, value), count in sorted(counter.items())]
        add("quantity_mismatch", {"unmatched_source": printable(before - after),
                                  "unmatched_target": printable(after - before)})
    if bound_counter(source) != bound_counter(target):
        add("explicit_size_bound_mismatch", "Explicit supported length bound differs after unit normalization.")
    source_dates, target_dates = dates(source), dates(target)
    if Counter(d["canonical"] for d in source_dates if not d["ambiguous"]) != Counter(
            d["canonical"] for d in target_dates if not d["ambiguous"]):
        add("date_mismatch", "Unambiguous date multiset differs; ordering alone is ignored.")
    if (any(d["ambiguous"] and d["raw"] not in target for d in source_dates) or
            any(d["ambiguous"] and d["raw"] not in source for d in target_dates)):
        add("ambiguous_date_reinterpreted", "Ambiguous source/target date is not preserved verbatim.")
    if Counter(y for _, _, y in calendar_years(source)) != Counter(y for _, _, y in calendar_years(target)):
        add("calendar_year_mismatch", "Explicit calendar-year multiset differs.")
    original_prior = Counter(prior_roles(source))
    if original_prior and original_prior != Counter(prior_roles(target)):
        add("prior_measurement_mismatch", "Explicit prior measurement marker/value differs after length conversion.")
    if NEG.search(negation_text(source)) and not NEG.search(negation_text(target)):
        # The inherited helper neutralizes grammatical negation in plain
        # atelectasis descriptions and in size comparisons / cannot-exclude.
        add("explicit_negation_marker_missing", "Source has an explicit recognized negation, target has none.")
    for name, (source_pattern, target_pattern) in UNCERTAINTY.items():
        if re.search(source_pattern, source, re.I) and not re.search(
                source_pattern + "|" + target_pattern, target, re.I):
            add("uncertainty_" + name + "_missing", "Recognized source uncertainty marker family is absent from target.")
    original_sides, output_sides = side_set(source), side_set(target)
    if len(original_sides) == 1 and original_sides != output_sides:
        add("exclusive_laterality_mismatch", {"source_sides": sorted(original_sides),
                                              "target_sides": sorted(output_sides)})
    return issues


def screen_row(row):
    for name in ("id", "source", "report", "reference_zh"):
        if not isinstance(row.get(name), str):
            raise ValueError(f"Required string field missing: {name}")
    if type(row.get("teacher_accepted")) is not bool:
        raise ValueError("teacher_accepted must be an explicit boolean")
    report, target = row["report"], row["reference_zh"]
    details = risk_details(report, target)
    flags = sorted({d["code"] for d in details})
    status = ("teacher_format_rejected" if not row["teacher_accepted"] else
              "risk_flagged" if flags else "no_known_rule_flag")
    result = {"id": row["id"], "source_sha256": sha256_text(report),
              "source_length": len(report), "target_length": len(target),
              "teacher_accepted": row["teacher_accepted"], "flags": flags,
              "screen_status": status, "features": features(report, row["source"])}
    return result, {"id": row["id"], "source_sha256": result["source_sha256"],
                    "source": report, "target": target, "issues": details}


def distribution(rows):
    keys = ("source", "length_bin", "quantity_presence", "negation_presence",
            "uncertainty_presence", "temporal_presence")
    return {key: {str(value).lower() if isinstance(value, bool) else str(value):
                  {"count": count, "proportion": round(count / len(rows), 8) if rows else 0.0}
                  for value, count in sorted(Counter(r["features"][key] for r in rows).items())}
            for key in keys}


def distribution_shift(before, after):
    a, b = distribution(before), distribution(after)
    return {key: {value: {"before_count": a[key].get(value, {}).get("count", 0),
                          "after_count": b[key].get(value, {}).get("count", 0),
                          "before_proportion": a[key].get(value, {}).get("proportion", 0.0),
                          "after_proportion": b[key].get(value, {}).get("proportion", 0.0),
                          "percentage_point_change": round(100 * (
                              b[key].get(value, {}).get("proportion", 0.0) -
                              a[key].get(value, {}).get("proportion", 0.0)), 6)}
                  for value in sorted(set(a[key]) | set(b[key]))} for key in a}


def summarize_rows(rows):
    accepted = [r for r in rows if r["teacher_accepted"]]
    eligible = [r for r in accepted if not r["flags"]]
    return {"counts": {"all_training_rows": len(rows), "teacher_format_accepted": len(accepted),
                       "teacher_format_rejected": len(rows) - len(accepted),
                       "accepted_with_risk_flag": len(accepted) - len(eligible),
                       "accepted_no_known_rule_flag": len(eligible), "new_human_reviews": 0},
            "screen_status_counts": dict(sorted(Counter(r["screen_status"] for r in rows).items())),
            "flag_counts_all_rows": dict(sorted(Counter(f for r in rows for f in r["flags"]).items())),
            "flag_counts_teacher_accepted": dict(sorted(Counter(f for r in accepted for f in r["flags"]).items())),
            "all_rows_distribution": distribution(rows),
            "teacher_accepted_before_filter_distribution": distribution(accepted),
            "teacher_accepted_no_known_rule_flag_distribution": distribution(eligible),
            "before_after_distribution_shift": distribution_shift(accepted, eligible)}


def calibration_report(pairs, split):
    """Score known synthetic edits, never clinician correctness labels."""
    selected = [p for p in pairs if p["split"] == split]
    outcomes = []
    for pair in selected:
        for kind in ("control", "perturbed"):
            issues = risk_details(pair["source"], pair[kind + "_target"])
            outcomes.append({"pair_id": pair["pair_id"], "family": pair["family"],
                             "category": pair["category"], "candidate": kind,
                             "known_synthetic_perturbation": pair[kind + "_known_synthetic_perturbation"],
                             "flagged": bool(issues), "flags": sorted({i["code"] for i in issues})})

    def counts(items):
        positives = [r for r in items if r["known_synthetic_perturbation"]]
        negatives = [r for r in items if not r["known_synthetic_perturbation"]]
        tp = sum(r["flagged"] for r in positives)
        fp = sum(r["flagged"] for r in negatives)
        return {"known_perturbations": len(positives), "flagged_known_perturbations": tp,
                "missed_known_perturbations": len(positives) - tp,
                "control_candidates": len(negatives), "flagged_controls": fp,
                "unflagged_controls": len(negatives) - fp,
                "synthetic_perturbation_detection_rate": tp / len(positives) if positives else None,
                "synthetic_control_false_flag_rate": fp / len(negatives) if negatives else None}
    summary = {"split": split, "pair_count": len(selected), "candidate_count": len(outcomes),
               "families": sorted({p["family"] for p in selected}), "counts": counts(outcomes),
               "by_category": {c: counts([r for r in outcomes if r["category"] == c])
                               for c in sorted({p["category"] for p in selected})},
               "flag_counts": dict(sorted(Counter(f for r in outcomes for f in r["flags"]).items())),
               "missed_pair_ids": [r["pair_id"] for r in outcomes if r["known_synthetic_perturbation"] and not r["flagged"]],
               "false_flag_pair_ids": [r["pair_id"] for r in outcomes if not r["known_synthetic_perturbation"] and r["flagged"]]}
    return summary, outcomes
