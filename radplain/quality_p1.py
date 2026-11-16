"""Conservative P1 source/target risk signals, developed on OLD material only.

clear_issue: a narrow explicit inconsistency or unsupported assertion with a
unique local correspondence. review_needed: weaker lexical evidence, omission,
ambiguity, or unresolved correspondence; it is NOT an automatic deletion rule.
no_supported_issue_found is not factual approval. All judgments remain rules,
not human/clinical labels. Exact evidence spans belong in local ignored outputs.
"""
from collections import Counter
from datetime import date
import hashlib
import re

from .common import fingerprint
from .fidelity import _QUANTITY
from .guard_v2 import NEG, negation_text
from .quality_p0 import UNCERTAINTY, quantity_counter, bound_counter
from .quantity_v2 import calendar_years


VERSION = "p1-conservative-risk-dev1"
MONTH_NAMES = ("january", "february", "march", "april", "may", "june",
               "july", "august", "september", "october", "november", "december")
MONTHS = {variant: i for i, name in enumerate(MONTH_NAMES, 1)
          for variant in (name, name[:3])}
MONTHS["sept"] = 9
MONTH = "(?:" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?"
PLACEHOLDER = re.compile(r"\[\s*date\s*\]|\bXXXX\b|日期不详|日期缺失", re.I)
CLAUSE_BOUNDARY = re.compile(r"[。；;\n]+|[.!?](?=\s|$)")

LIMITATIONS = [
    "No supported issue found does not establish source faithfulness, clinical correctness or completeness.",
    "review_needed is a retained review queue, not an automatic rejection or confirmed error label.",
    "Rules were developed on old 18 training-target audits and 10 old pilot-development reports; these are development evidence only.",
    "New human reviews remain zero; neither prior project-AI notes nor rule outputs are human gold.",
    "Date formats and terminology lexicons are finite; ambiguity and source errors are not resolved by this screen.",
    "Clear substitutions require unique local correspondence; multi-finding reports are deliberately downgraded to review_needed.",
    "Explicit quantity changes cover mm/cm and day/week/month/year; organ/value binding and implicit measurements remain unchecked.",
    "Normality, resolution and missing device endpoints are narrow lexical checks; plausible synonyms and cross-sentence references can be missed.",
    "Absence of a target word is not generally proof of omission; most such signals require review.",
]


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _evidence(text, start, end):
    return {"start": start, "end": end, "text": text[start:end]}


def _clause(text, match):
    start, end = 0, len(text)
    for boundary in CLAUSE_BOUNDARY.finditer(text):
        if boundary.end() <= match.start():
            start = boundary.end()
        elif boundary.start() >= match.end():
            end = boundary.end()
            break
    return _evidence(text, start, end)


def _valid(y, m, d):
    try:
        date(y, m, d)
        return True
    except ValueError:
        return False


def date_mentions(text):
    """Recognize supported full/partial dates with exact spans, longest first.

    Numeric day/month dates retain BOTH plausible interpretations. A month alone
    needs explicit calendar/study context, so modal 'may' and 'seven months'
    are not interpreted as dates.
    """
    found = []

    def add(match, values, kind):
        if any(match.start() < x["end"] and match.end() > x["start"] for x in found):
            return
        values = sorted(set(values), key=str)
        found.append({**_evidence(text, match.start(), match.end()), "values": values,
                      "kind": kind, "ambiguous": len(values) != 1})

    for m in re.finditer(r"(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)|"
                         r"(?<!\d)(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text):
        parts = m.groups()[:3] if m[1] else m.groups()[3:]
        y, mo, d = map(int, parts)
        add(m, [(y, mo, d)] if _valid(y, mo, d) else [], "full")
    for m in re.finditer(r"(?<!\d)(\d{1,2})\s*[/ .-]\s*(\d{1,2})\s*[/ .-]\s*(\d{4})(?!\d)", text):
        a, b, y = map(int, m.groups())
        add(m, [(y, mo, d) for mo, d in ((a, b), (b, a)) if _valid(y, mo, d)], "full")
    for m in re.finditer(rf"\b(\d{{1,2}})\s+({MONTH})\s+(\d{{4}})\b", text, re.I):
        d, mo, y = int(m[1]), MONTHS[m[2].lower().rstrip(".")], int(m[3])
        add(m, [(y, mo, d)] if _valid(y, mo, d) else [], "full")
    for m in re.finditer(rf"\b({MONTH})\s+(?:(\d{{1,2}})(?:st|nd|rd|th)?[,]?\s+)?(\d{{4}})\b", text, re.I):
        mo, y, d = MONTHS[m[1].lower().rstrip(".")], int(m[3]), int(m[2]) if m[2] else None
        values = [(y, mo, d)] if d is None or _valid(y, mo, d) else []
        add(m, values, "full" if d is not None else "month_year")
    for m in re.finditer(r"(?<!\d)((?:18|19|20|21)\d{2})\s*年\s*(\d{1,2})\s*月(?:份)?", text):
        y, mo = map(int, m.groups())
        add(m, [(y, mo, None)] if 1 <= mo <= 12 else [], "month_year")
    for m in re.finditer(rf"\b({MONTH})(?=\s+(?:study|scan|examination|exam|radiograph|x-ray|comparison)\b)", text, re.I):
        add(m, [(None, MONTHS[m[1].lower().rstrip(".")], None)], "month")
    for m in re.finditer(r"(?<![\d个])(\d{1,2})\s*月(?:份)?(?=\s*(?:的|检查|影像|研究|扫描|复查))", text):
        mo = int(m[1])
        add(m, [(None, mo, None)] if 1 <= mo <= 12 else [], "month")
    for start, end, year in calendar_years(text):
        if not any(start < x["end"] and end > x["start"] for x in found):
            found.append({**_evidence(text, start, end), "values": [(int(year), None, None)],
                          "kind": "year", "ambiguous": False})
    return sorted(found, key=lambda x: x["start"])


def _masked_dates(text, mentions):
    chars = list(text)
    for mention in mentions:
        chars[mention["start"]:mention["end"]] = " " * (mention["end"] - mention["start"])
    return "".join(chars)


def _compatible(a, b):
    return all(x is None or y is None or x == y for x, y in zip(a, b))


def _unparsed_calendar_markers(text, mentions):
    masked = _masked_dates(text, mentions)
    # A partially recognized date range must not masquerade as a unique date.
    # Exclude bare modal 'may'; a recognized May date is already masked.
    names = "|".join(x for x in MONTHS if x != "may")
    return bool(re.search(rf"\b(?:{names})\b|\d\s*(?:年|月)(?!后|前|内)", masked, re.I))


TERMS = (
    # Only the listed suspicious substitutions are checked; this is not an
    # ontology or a test that every correct answer must contain technical jargon.
    ("consolidation", r"\bconsolidations?\b|实变", r"实变|肺泡.{0,12}(?:充满|填满)|肺.{0,8}(?:变实|变致密)",
     ((r"结节", r"\bnodules?\b|结节"), (r"肺炎", r"\bpneumonia\b|肺炎"),
      (r"感染", r"\binfect\w*\b|感染"), (r"(?:组织|肺).{0,3}增厚", r"\bthicken\w*\b|增厚"))),
    ("infiltrate", r"\binfiltrat(?:e|es|ion|ions)\b|浸润", r"浸润|渗出|阴影|影子|密度增高",
     ((r"感染", r"\binfect\w*\b|感染"), (r"肺炎", r"\bpneumonia\b|肺炎"),
      (r"结节", r"\bnodules?\b|结节"))),
)
FINDING_WORDS = re.compile(r"\b(?:consolidations?|infiltrat(?:e|es|ion|ions)|nodules?|pneumonia|"
                           r"effusion|pneumothorax|atelectasis|opacity|opacities|infection|mass|emphysema|COPD)\b", re.I)
DEVICES = (
    ("endotracheal_tube", r"\b(?:endo|oro)tracheal\s+(?:tube|intubation)\b|气管插管", r"气管.{0,3}(?:插管|导管)"),
    ("nasogastric_tube", r"\bnasogastric\s+tube\b|鼻胃管|胃管", r"鼻胃管|胃管|胃肠管"),
    ("central_line", r"\bcentral\s+(?:line|venous\s+catheter)\b|中心静脉导管", r"中心静脉.{0,3}(?:管|置管)|中心导管|中心静脉通路"),
    ("pacemaker", r"\bpacemakers?\b|起搏器", r"(?:心脏)?起搏器"),
    ("vp_shunt", r"\bventriculoperitoneal\s+shunt(?:\s+catheter)?\b|脑室腹腔分流管", r"脑室.{0,2}腹腔.{0,3}分流(?:管)?"),
)


def screen_pair(source, target):
    if not isinstance(source, str) or not isinstance(target, str):
        raise TypeError("Source and target must be strings")
    issues = []

    def add(code, level, reason, source_evidence=None, target_evidence=None, **detail):
        def evidence(value, text):
            if value is None:
                return [_evidence(text, 0, len(text))]
            if isinstance(value, list):
                return [{k: x[k] for k in ("start", "end", "text")} for x in value]
            if isinstance(value, dict):
                return [{k: value[k] for k in ("start", "end", "text")}]
            return [_evidence(text, value.start(), value.end())]
        issues.append({"code": code, "level": level, "reason": reason,
                       "source_evidence": evidence(source_evidence, source),
                       "target_evidence": evidence(target_evidence, target), "detail": detail})

    if not target.strip():
        add("empty_target", "clear_issue", "目标文本为空，未完成改写。")
    sd, td = date_mentions(source), date_mentions(target)
    # Equality handles translated month names, date order and partial dates.
    sk = Counter(tuple(x["values"][0]) for x in sd if not x["ambiguous"])
    tk = Counter(tuple(x["values"][0]) for x in td if not x["ambiguous"])
    if any(x["ambiguous"] for x in sd + td):
        for mention in sd:
            if not mention["ambiguous"] or mention["text"] in target:
                continue
            compatible = [t for t in td if any(_compatible(a, b) for a in mention["values"] for b in t["values"])]
            impossible = (len(sd) == len(td) == 1 and mention["values"] and td[0]["values"] and not compatible
                          and not _unparsed_calendar_markers(source, sd)
                          and not _unparsed_calendar_markers(target, td))
            add("ambiguous_date_incompatible" if impossible else "ambiguous_date_needs_review",
                "clear_issue" if impossible else "review_needed",
                "目标日期不属于源文数字日期任何可行的日月解释。" if impossible else "源文日期有歧义或格式无效；不能把某种解释当成已确认日期。",
                mention, td or None)
        if any(x["ambiguous"] and x["text"] not in source for x in td):
            add("target_date_ambiguous", "review_needed", "目标出现新的含糊或无效日期。", sd or None, td)
    if sk != tk:
        # One ambiguous source date handled above must not also become a false
        # exact-date discrepancy merely because its interpretation was selected.
        if not any(x["ambiguous"] for x in sd + td):
            incompatible = (len(sd) == len(td) == 1 and not PLACEHOLDER.search(source)
                            and not _unparsed_calendar_markers(source, sd)
                            and not _unparsed_calendar_markers(target, td)
                            and not _compatible(sd[0]["values"][0], td[0]["values"][0]))
            add("explicit_date_changed" if incompatible else "date_information_needs_review",
                "clear_issue" if incompatible else "review_needed",
                "唯一源日期与唯一目标日期的明确部分冲突。" if incompatible else "日期信息数量、精度或匹配关系不同，需要人工核对；没有按顺序强行配对。",
                sd or None, td or None)
        elif not any(i["code"].startswith(("ambiguous_date_", "target_date_")) for i in issues):
            add("mixed_ambiguous_date_information", "review_needed",
                "含糊日期虽原样保留，其他日期信息仍有未匹配项；不能假定唯一对应。", sd or None, td or None)
    sq = quantity_counter(_masked_dates(source, sd))
    tq = quantity_counter(_masked_dates(target, td))
    if sq != tq:
        one_each = sum(sq.values()) == sum(tq.values()) == 1
        same_kind = one_each and next(iter(sq))[0] == next(iter(tq))[0]
        # A single explicit value can be compared without guessing organ roles.
        clear = same_kind and not PLACEHOLDER.search(source)
        add("explicit_quantity_changed" if clear else "quantity_information_needs_review",
            "clear_issue" if clear else "review_needed",
            "唯一同维度显式量值在单位换算后仍不同。" if clear else "显式数值未匹配；多值对象关联、遗漏或新增需要复核。",
            detail_source_quantities=[(k, str(v), n) for (k, v), n in sq.items()],
            detail_target_quantities=[(k, str(v), n) for (k, v), n in tq.items()])
    if bound_counter(source) != bound_counter(target):
        add("size_bound_needs_review", "review_needed", "明确长度界限表达不同；需核对比较关系与对象。")
    sm = NEG.search(negation_text(source))
    if sm and not NEG.search(negation_text(target)):
        # negation_text may change offsets, so preserve the full original text.
        add("negation_marker_missing", "review_needed", "源文有显式否定而目标未识别到；同义表达和作用范围需复核。")
    for name, (sp, tp) in UNCERTAINTY.items():
        sm = re.search(sp, source, re.I)
        if sm and not re.search(sp + "|" + tp, target, re.I):
            add("uncertainty_" + name + "_missing", "review_needed", "明确不确定性词族未匹配；需核对语气和范围。", _clause(source, sm))

    for name, sp, expected, replacements in TERMS:
        source_matches = list(re.finditer(sp, source, re.I))
        if not source_matches or re.search(expected, target, re.I):
            continue
        for bad_target, bad_source in replacements:
            tm = re.search(bad_target, target, re.I)
            if not tm or re.search(bad_source, source, re.I):
                continue
            sc, tc = _clause(source, source_matches[0]), _clause(target, tm)
            # Strict single-finding, single-clause cases only. Longer/mixed
            # reports get a review signal, never a confident word-bag verdict.
            unique = (len(source_matches) == 1 and len(FINDING_WORDS.findall(source)) == 1
                      and source.strip() == sc["text"].strip() and target.strip() == tc["text"].strip())
            add(name + "_possible_substitution", "clear_issue" if unique else "review_needed",
                "单一发现的关键术语被不等价概念替代，源文未出现该新概念。" if unique else "关键术语未匹配且出现可疑替代概念；多发现/同义表达需人工核对。", sc, tc)

    for name, sp, tp in DEVICES:
        sms = list(re.finditer(sp, source, re.I))
        tms = list(re.finditer(tp, target, re.I))
        if not sms:
            continue
        sc = _clause(source, sms[0])
        tc = _clause(target, tms[0]) if tms else None
        if not tms:
            add(name + "_not_matched", "review_needed", "源文器械未找到支持的目标同义词；可能遗漏或发生术语变化。", sc)
            continue
        unique = len(sms) == len(tms) == 1
        if name == "pacemaker":
            chamber = re.search(r"\b(?:bicameral|dual[- ]chamber|two[- ]chamber)\b|双腔", sc["text"], re.I)
            plural = re.search(r"(?:两|二|2)\s*(?:个|台|枚)\s*(?:心脏)?起搏器|双心起搏器", tc["text"])
            source_multiple = re.search(r"\b(?:two|2)\s+(?:cardiac\s+)?pacemakers\b|(?:两|2)个.{0,3}起搏器", source, re.I)
            if chamber and plural and not source_multiple:
                # A parenthetical alias can repeat 'pacemaker' in the SAME
                # count assertion; multiple separate target clauses stay weak.
                target_clauses = {(_clause(target, m)["start"], _clause(target, m)["end"]) for m in tms}
                count_unique = len(sms) == 1 and len(target_clauses) == 1
                add("pacemaker_chambers_as_device_count", "clear_issue" if count_unique else "review_needed",
                    "双腔是器械结构，目标却表述为两个起搏器；源文未提供两台器械。", sc, tc)
        tip = re.search(r"\b(?:distal\s+(?:end|tip)|tip)\b|远端|尖端|末端", sc["text"], re.I)
        target_tip = re.search(r"远端|尖端|末端|端部|末梢|头端|管尖", tc["text"])
        if tip and not target_tip:
            add(name + "_endpoint_not_matched", "review_needed", "原文定位的是器械端点，目标未保留可识别的端点限定。", sc, tc)
        for anchor_en, anchor_zh in ((r"\bcarina\b", r"隆突|气管分叉"), (r"\bdiaphragm\b", r"膈肌|横膈")):
            if not (re.search(anchor_en, sc["text"], re.I) and re.search(anchor_zh, tc["text"])):
                continue
            above = bool(re.search(r"\babove\b|上方|之上|以上", sc["text"], re.I))
            below = bool(re.search(r"\bbelow\b|下方|之下|以下", sc["text"], re.I))
            target_above = bool(re.search(r"上方|之上|以上", tc["text"]))
            target_below = bool(re.search(r"下方|之下|以下", tc["text"]))
            if above != below and target_above != target_below and above != target_above:
                add(name + "_position_direction_reversed", "clear_issue" if unique else "review_needed",
                    "同一器械与同一解剖参照的明确上下关系相反。", sc, tc)
        normal = re.search(r"位置.{0,4}(?:正常|良好|正确)|(?:放置|放得|置入).{0,4}(?:正确|良好)", tc["text"])
        source_location_or_judgment = re.search(
            r"\b(?:tip|end|position\w*|project\w*|terminat\w*|within|below|above|extend\w*|stable|unchanged|normal|appropriate|correct)\b|"
            r"尖端|末端|位置|位于|正常|稳定|满意", sc["text"], re.I)
        simple_listing = len(sc["text"].split()) <= 7 and not re.search(r"\band\b|[,;；，]", sc["text"], re.I)
        if normal and not source_location_or_judgment:
            add(name + "_unsupported_normal_position", "clear_issue" if unique and simple_listing else "review_needed",
                "源文只列出器械，目标增加其位置正常/正确的判断。", sc, tc)

    resolution = re.compile(r"\b(?:interval\s+|radiological\s+)?resolution\s+of\b|"
                            r"\b(?:has|have|had)(?:\s+completely)?\s+resolved\b|"
                            r"\b(?:fully|completely)\s+resolved\b|(?:已经|已|完全|均已).{0,4}(?:消退|消失|吸收)", re.I)
    for sm in resolution.finditer(source):
        prefix = source[max(0, sm.start() - 35):sm.start()]
        if re.search(r"\b(?:no|not|without|incomplete|partial|assess|assessing|evaluate|evaluating)\s*(?:complete\s*)?$|未|尚未", prefix, re.I):
            continue
        if not re.search(r"消退|消失|完全吸收|已吸收|恢复正常|已恢复|不再可见|不再显示|已经没有|已无|"
                         r"(?:现在|目前|已|已经).{0,10}(?:看不到|看不见)|(?:看不到|看不见)了", target):
            add("resolution_not_preserved", "review_needed", "源文明确描述消退/消失，目标没有匹配到该变化；改善不等于消退。", _clause(source, sm))
            break
    sm = re.search(r"\bcardiothoracic\s+(?:index|ratio)\b.{0,20}\bnormal\b", source, re.I)
    tm = re.search(r"心脏.{0,12}(?:正常|没有.{0,4}异常|未见.{0,4}异常)", target)
    if sm and tm and not re.search(r"心胸比|心胸比例|心脏.{0,4}(?:大小|尺寸)|心影.{0,4}大小", target):
        add("cardiothoracic_normal_scope", "review_needed", "心胸比正常可能被扩大为心脏整体正常，需确认目标表达范围。", _clause(source, sm), _clause(target, tm))

    status = ("clear_issue" if any(i["level"] == "clear_issue" for i in issues) else
              "review_needed" if issues else "no_supported_issue_found")
    return {"version": VERSION, "status": status, "flags": sorted({i["code"] for i in issues}),
            "issues": issues, "human_reviewed": False,
            "source_sha256": _sha(source), "normalized_source_sha256": fingerprint(source),
            "target_sha256": _sha(target),
            "source_limitations": ["placeholder_or_redaction"] if PLACEHOLDER.search(source) else [],
            "scope": "lexical_source_target_risk_not_clinical_correctness"}


def screen_row(row):
    """Preserve historical metadata while exposing an explicit screening status."""
    if type(row.get("teacher_accepted")) is not bool:
        raise ValueError("teacher_accepted must be an explicit boolean")
    return {"id": row["id"], "source": row["source"],
            "teacher_accepted": row["teacher_accepted"],
            "source_length": len(row["report"]), "target_length": len(row["reference_zh"]),
            **screen_pair(row["report"], row["reference_zh"])}
