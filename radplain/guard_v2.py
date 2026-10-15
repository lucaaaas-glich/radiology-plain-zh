"""Auditable lexical checks, NOT a medical verifier or semantic certificate.

Checks deliberately require numeric order and the original units. Some valid
paraphrases/conversions are flagged; unsupported paraphrases can escape checks.
The caller must report coverage, keep the source, and never call a pass 'correct'.
"""
import re
from decimal import Decimal
from .common import chinese_ratio
from .fidelity import numeric_review, _number

VERSION = 'lexical-guard-v2-dev2'
CN = '零〇一二两三四五六七八九十百点半'
UNIT = r'mm|cm|millimet(?:er|re)s?|centimet(?:er|re)s?|毫米|厘米|months?|个月|月|years?|年|weeks?|星期|周|days?|天'
ORDINALS = dict(zip(('first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth').split(), range(1, 13)))
NUMBERS = re.compile(rf'\d+(?:\.\d+)?|第([{CN}]+)|([{CN}]+)(?=\s*(?:{UNIT}))|\b(' + '|'.join(ORDINALS) + r')\b', re.I)


def numeric_sequence(text):
    """Preserve multiplicity/order, unlike v1's unordered quantity set."""
    values = []
    for m in NUMBERS.finditer(text):
        try:
            if m[3]:
                n = Decimal(ORDINALS[m[3].lower()])
            else:
                n = _number(m[1] or m[2] or m.group())
            values.append(str(n.normalize()))
        except (ArithmeticError, IndexError, KeyError, ValueError):
            values.append(m.group())
    return values


def spans(report):
    """Keep exact source offsets; decimals are not sentence boundaries.

    This is punctuation segmentation, not clinical discourse parsing. Abbreviated
    prose and references across sentences may still need the supplied context.
    """
    boundaries = list(re.finditer(r'[。！？!?]+|\.(?=\s|$)|[\r\n]+', report))
    result, start = [], 0
    for m in boundaries:
        # Common abbreviations are not boundaries.
        if m.group() == '.' and re.search(r'\b(?:Dr|Mr|Ms|Mrs|e\.g|i\.e|vs)\.$', report[:m.end()], re.I):
            continue
        end = m.end()
        a, b = start, end
        while a < b and report[a].isspace():
            a += 1
        while b > a and report[b-1].isspace():
            b -= 1
        if a < b:
            result.append({'source': report[a:b], 'start': a, 'end': b})
        start = end
    a, b = start, len(report)
    while a < b and report[a].isspace():
        a += 1
    while b > a and report[b-1].isspace():
        b -= 1
    if a < b:
        result.append({'source': report[a:b], 'start': a, 'end': b})
    return result


def administrative_instruction(text):
    """Only flag obvious command text; keep it visible as excluded source."""
    return bool(re.search(r'ignore\s+(?:all\s+)?(?:previous\s+instructions|(?:the|this)\s+report)|'
                          r'忽略.{0,12}(?:报告|指令)|(?:administrative|行政).{0,80}(?:say|prescribe|告知|确诊)', text, re.I))


FEATURES = [
    ('left', r'\bleft\b|左', r'左'),
    ('right', r'\bright\b|右', r'右'),
    ('bilateral', r'\bbilateral\b|\bboth\b|\bbibasa[lr]\b|双|两侧', r'双|两侧|左右|两肺|两肾|两边'),
    ('indeterminate', r'\bindeterminate\b|性质待定|不能确定|无法确定|未能确定|尚不能确认', r'待定|未定|不确定|未确定|未明确|不能确定|无法确定|不明确|不能确认|不清楚'),
    ('cannot_exclude', r'cannot\s+(?:be\s+)?(?:excluded|exclude|rule\s+out)|不能.{0,3}排除|无法排除', r'不能.{0,3}排除|无法排除|不排除'),
    ('possible', r'\bmay\b|\bmight\b|\bcould\b|\bpossible\b|\bsuspicious\b|可能|可疑', r'可能|可疑|怀疑|考虑|也许'),
    ('stable', r'\bstable\b|\bunchanged\b|稳定|无变化|未变', r'稳定|未变|无变化|没有变化|没有改变|没变|相同|一样'),
    ('improved', r'\bdecreased\b|\bimproved\b|\breduced\b|减少|缩小|吸收|好转', r'减少|缩小|变小|减轻|变少|好转|改善|吸收'),
    ('increased', r'\bincreased\b|\benlarged\b|增多|增大', r'增多|增大|增加|变大|扩大|变多|略大|稍大|更大'),
    ('residual', r'\bresidual\b|\bremains?\b|not\s+(?:completely|fully)\s+resolved|残留|仍|尚未完全', r'残留|仍|还有|还未|没有完全|尚未|依然'),
    ('recommendation', r'\brecommended\b|\badvised\b|建议', r'建议|推荐'),
    ('prior_comparison', r'\bpreviously\b|\bprevious\b|\bprior\b|既往|较前|以前|前次', r'既往|此前|前次|之前|以前|上次|原来|过去|较前|原先'),
    ('carina', r'\bcarina\b|隆突|气管分叉', r'隆突|气管分叉|气管.{0,4}分成|气管.{0,4}分支'),
    ('above', r'\babove\b|上方', r'上方|以上|之上'),
    ('below', r'\bbelow\b|下方|膈下', r'下方|以下|膈下|之下'),
    ('lung_base', r'\bbibas\w*\b|\blung\s+bases?\b|肺底', r'肺底|底部|下部|下肺|肺下'),
    ('upper_lobe', r'\bupper\s+lobe\b|肺上叶', r'上叶|上部|上肺|肺上'),
    ('lower_lobe', r'\blower\s+lobe\b|肺下叶', r'下叶|下部|下肺|肺下'),
    ('middle_lobe', r'\bmiddle\s+lobe\b|肺中叶', r'中叶|中部|中间'),
    ('limited_assessment', r'\blimit\w*\b|poorly\s+assessed|cannot\s+be\s+confirmed|not\s+included|限制|欠清|未能评价|未完全纳入', r'限制|受限|影响|不清|看不清|无法|不能|未能|没有.{0,4}拍|未.{0,4}拍|不在|不包括|未包括|未完全'),
    ('newness', r'\bnew\b|新发|新出现|新增', r'新'),
    ('negative_qualifier', r'no\s+(?:definite|obvious)|未见(?:明显|明确)', r'(?:未见|没有|未发现|看不到).{0,3}(?:明显|明确|清楚)'),
]

UNCERTAIN = re.compile(r'cannot\s+(?:be\s+)?(?:excluded|exclude|rule\s+out)|不能.{0,3}排除|无法排除|不排除', re.I)
NEG = re.compile(r'\bno\b|\bwithout\b|\bneither\b|未见|没有|未发现|不大|不宽|无(?:明显|新发|新出现|胸|积|气|急性|实变|异常)', re.I)

# These checks find loss of a common concept, not its full clinical relationship.
CONCEPTS = [
    ('pleural_effusion', r'pleural\s+effusion|胸腔积液', r'积液|肺.{0,12}(?:液体|积水)|胸.{0,12}(?:液体|积水)'),
    ('pneumothorax', r'pneumothorax|气胸', r'气胸|肺.{0,12}(?:空气|气体)|胸.{0,12}(?:空气|气体)'),
    ('atelectasis', r'atelectasis|肺不张', r'肺不张|肺.{0,12}(?:展开|张开|充气|塌陷|扩张)'),
    ('nodule', r'\bnodules?\b|结节', r'结节|小.{0,2}(?:团块|肿块|圆点|点状|硬块)'),
    ('malignancy', r'malignan\w*|cancer|恶性|肿瘤|癌', r'恶性|肿瘤|癌'),
    ('opacity', r'\bopacit\w*\b|阴影|斑片影', r'阴影|影子|斑片|变白|白影|模糊'),
    ('consolidation', r'\bconsolidation\b|实变', r'实变|致密|变密|变白'),
]


def prior_roles(text):
    """Limited prior-measurement recognition; no arbitrary temporal reasoning."""
    found = []
    pattern = rf'(?:previously|previous\s+(?:size|measurement)(?:\s+was)?|此前|前次|以前|原来|原先|之前|较前|上次)(?:\s|为|是|约|的|大小|测量|只有|仅有|只有约|结节)*?(\d+(?:\.\d+)?|[{CN}]+)\s*(mm|cm|毫米|厘米)'
    for m in re.finditer(pattern, text, re.I):
        try:
            found.append(str((_number(m[1]) * (10 if m[2].lower() in ('cm', '厘米') else 1)).normalize()))
        except (ValueError, KeyError, ArithmeticError):
            pass
    return found


def concept_negations(text, pattern):
    """Limited clause-prefix negation for common concepts; not a clinical parser."""
    result=[]
    for m in re.finditer(pattern, text, re.I):
        prefix=re.split(r'[。.;；，,!?]|\bbut\b|\bwhile\b',text[:m.start()],flags=re.I)[-1]
        result.append(bool(NEG.search(UNCERTAIN.sub('',prefix))))
    return result


def review(source, prediction, finish_reason='stop'):
    issues = []
    def add(code, detail):
        issues.append({'code': code, 'detail': detail})
    if not prediction.strip():
        add('empty', '没有完整解释')
    elif chinese_ratio(prediction) < .65:
        add('language', '中文比例不足')
    if finish_reason != 'stop':
        add('incomplete', '生成未正常结束，可能不完整')
    original, output = numeric_sequence(source), numeric_sequence(prediction)
    if original != output:
        add('numeric_sequence', f'数值数量或顺序不一致：原文{original}，改写{output}；请保留原文数值顺序，不做计算')
    q = numeric_review(source, prediction)
    if q['warnings']:
        add('quantity', '；'.join(q['warnings']))
    original_prior, output_prior = prior_roles(source), prior_roles(prediction)
    if original_prior and original_prior != output_prior:
        add('prior_measurement', f'以前的测量值未按原文对应：{original_prior}；请明确“以前为…”')
    src_neg = bool(NEG.search(UNCERTAIN.sub('', source)))
    out_neg = bool(NEG.search(UNCERTAIN.sub('', prediction)))
    if src_neg != out_neg:
        add('negation_presence', '否定表达可能丢失或新增；请逐个核对被否定的发现')
    for name, src, target in FEATURES:
        present, expressed = bool(re.search(src, source, re.I)), bool(re.search(target, prediction, re.I))
        if present and not expressed:
            add(name, '请保留原文的' + name + '信息')
        # Some features may be legitimate rewordings; enforce additions only for sides.
        if name in ('left', 'right', 'bilateral') and expressed and not present:
            # Bilateral in source licenses left/right wording.
            bilateral_source = bool(re.search(FEATURES[2][1], source, re.I))
            if not (name in ('left', 'right') and bilateral_source):
                add('added_' + name, '出现原句没有的左右侧描述')
    for name, src, target in CONCEPTS:
        if re.search(src, source, re.I) and not re.search(target, prediction, re.I):
            add('missing_' + name, '原句中的' + name + '可能被遗漏')
        src_states=concept_negations(source,src)
        out_states=concept_negations(prediction,target)
        # Only compare one occurrence on each side: repeated findings and scope
        # across clauses cannot be safely resolved by this lexical heuristic.
        if len(src_states)==len(out_states)==1 and src_states!=out_states:
            add('negation_'+name, name+'的否定关系可能改变')
    if re.search(r'建议|推荐|需要.{0,5}(?:检查|治疗|复查)|应当|应该', prediction) and not re.search(r'recommend|advis|follow.up|repeat|建议|复查|随访|需', source, re.I):
        add('added_advice', '出现原句没有的建议')
    if administrative_instruction(prediction):
        add('instruction_echo', '输出含疑似行政指令')
    return {'version': VERSION, 'scope': 'limited_lexical_checks_not_clinical_validation',
            'issues': issues, 'no_flags': not issues, 'numeric_overlap': q}
