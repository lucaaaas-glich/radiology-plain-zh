"""Conservative number/unit overlap flags, not a medical or semantic verifier.

Added after pilot_v1 inspection. Never changes generated text or its raw scores.
It cannot verify relationships, negation, laterality, anatomy or clinical meaning.
"""
import re
from decimal import Decimal

_DIGITS = dict(zip("零〇一二两三四五六七八九", [0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9]))
_NUMBER = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百点半]+)"
_UNITS = r"(?:millimet(?:er|re)s?|centimet(?:er|re)s?|mm|cm|毫米|厘米|months?|个月|月|years?|年|weeks?|星期|周|days?|天)"
_QUANTITY = re.compile(rf"(?P<number>{_NUMBER})\s*(?P<unit>{_UNITS})(?![a-zA-Z])", re.I)


def _number(text):
    if text == "半":
        return Decimal("0.5")
    if text[0].isdigit() and text[0].isascii():
        return Decimal(text)
    if "点" in text:
        integer, fraction = text.split("点", 1)
        return _number(integer) + Decimal("0." + "".join(str(_DIGITS[c]) for c in fraction))
    if not any(c in text for c in "十百"):
        return Decimal("".join(str(_DIGITS[c]) for c in text))
    result, current = 0, 0
    for c in text:
        if c in _DIGITS:
            current = _DIGITS[c]
        else:
            result += (current or 1) * {"十": 10, "百": 100}[c]
            current = 0
    return Decimal(result + current)


def quantities(text):
    found = {}
    for match in _QUANTITY.finditer(text):
        try:
            value = _number(match["number"])
        except (ArithmeticError, IndexError, KeyError, ValueError):
            continue
        unit = match["unit"].lower()
        if unit in ("mm", "毫米") or unit.startswith("millimet"):
            kind, scale = "length_mm", 1
        elif unit in ("cm", "厘米") or unit.startswith("centimet"):
            kind, scale = "length_mm", 10
        elif unit in ("年",) or unit.startswith("year"):
            kind, scale = "duration_months", 12
        elif unit in ("月", "个月") or unit.startswith("month"):
            kind, scale = "duration_months", 1
        elif unit in ("周", "星期") or unit.startswith("week"):
            kind, scale = "duration_days", 7
        else:
            kind, scale = "duration_days", 1
        found.setdefault((kind, value * scale), match.group().strip())
    return found


def numeric_review(source, prediction):
    original, output = quantities(source), quantities(prediction)
    extra = [v for k, v in output.items() if k not in original]
    missing = [v for k, v in original.items() if k not in output]
    warnings = []
    if extra:
        warnings.append("输出出现原文未直接列出的数值：" + "、".join(extra))
    if missing:
        warnings.append("原文数值未在输出中匹配到：" + "、".join(missing))
    return {"scope": "number_unit_overlap_only_not_semantic_or_medical_validation",
            "extra_output_quantities": extra, "unmatched_source_quantities": missing,
            "warnings": warnings}


def print_numeric_warnings(review):
    if review["warnings"]:
        print("[数值需核对] " + "；".join(review["warnings"]) + "。请对照原文，不要直接采用这段解释。")
