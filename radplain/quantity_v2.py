"""Normalize common spelled durations and dates before lexical quantity checks."""
import re
from datetime import date
from decimal import Decimal
from .fidelity import _number, numeric_review

CN='零〇一二两三四五六七八九十百点'
WORDS={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
       'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,
       'eighteen':18,'nineteen':19,'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,
       'seventy':70,'eighty':80,'ninety':90}
EN='(?:'+'|'.join(WORDS)+')'
UNITS=r'(?:millimet(?:er|re)s?|centimet(?:er|re)s?|mm|cm|months?|years?|weeks?|days?)'
DATES=re.compile(r'(?<!\d)\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d)|(?<!\d)\d{1,2}([/ -])\d{1,2}\1\d{4}(?!\d)|\d{4}年\d{1,2}月\d{1,2}日')
YEAR = r'(?:18|19|20|21)\d{2}'
EN_CALENDAR = re.compile(rf'\b(?:in|since|during|from|year)\s+({YEAR})\b(?!\s*(?:{UNITS})\b)', re.I)
CN_CALENDAR = re.compile(rf'(?<!\d)({YEAR})年(?![后前年余多来内间])')
DURATION_PREFIX = re.compile(r'(?:持续|历时|经过|长达|为期|共|累计|约|超过)\s*$')


def calendar_years(text):
    """Recognize explicit calendar context, never every four-digit number.

    This deliberately covers a narrow range of modern report years. Bare IDs,
    measurements and English '2015 years' must remain numeric quantities. Chinese
    duration cues keep an otherwise ambiguous YYYY年 out of calendar matching.
    Full dates are handled separately by dates().
    """
    masked = DATES.sub(lambda m: ' ' * len(m.group()), text)
    spans = []
    for m in EN_CALENDAR.finditer(masked):
        spans.append((m.start(1), m.end(1), m[1]))
    for m in CN_CALENDAR.finditer(masked):
        if not DURATION_PREFIX.search(masked[max(0, m.start()-8):m.start()]):
            spans.append((m.start(), m.end(), m[1]))
    return sorted(spans)


def dates(text):
    result=[]
    for m in DATES.finditer(text):
        values=list(map(int,re.findall(r'\d+',m.group())))
        a,b,c=values
        ambiguous=False
        try:
            if len(re.match(r'\d+',m.group()).group())==4:
                canonical=date(a,b,c).isoformat()
            elif a>12:
                canonical=date(c,b,a).isoformat()
            elif b>12:
                canonical=date(c,a,b).isoformat()
            else:
                ambiguous=True
                canonical=m.group()
        except ValueError:
            ambiguous=True
            canonical=m.group()
        result.append({'raw':m.group(),'canonical':canonical,'ambiguous':ambiguous})
    return result


def normalized_quantities_text(text):
    for start, end, year in reversed(calendar_years(text)):
        text=text[:start]+' [YEAR] '+text[end:]
    text=DATES.sub(' [DATE] ',text)
    text=re.sub(r'COVID[ -]?19','COVID',text,flags=re.I)
    text=re.sub(rf'(\d+(?:\.\d+)?|[{CN}]+)(?:周)?岁',r'\1年',text)
    text=re.sub(rf'\b({EN})-(?={UNITS}\b)',r'\1 ',text,flags=re.I)
    def convert_en(m):
        n=sum(WORDS[w.lower()] for w in re.findall(EN,m[1],re.I))
        if m[2]:
            n=Decimal(n)+Decimal('.5')
        return str(n)+' '+m[3]
    text=re.sub(rf'\b({EN}(?:[- ]{EN})?)(\s+and\s+a\s+half)?\s+({UNITS})\b',convert_en,text,flags=re.I)
    text=re.sub(rf'\bhalf\s+(?:a\s+)?({UNITS})\b',r'0.5 \1',text,flags=re.I)
    text=re.sub(rf'(\d+(?:\.\d+)?|[{CN}]+)(?:个)?半\s*(个月|月|年|天|周)',
                lambda m:str(_number(m[1])+Decimal('.5'))+m[2],text)
    text=re.sub(r'(\d+)-(?=(?:months?|years?|weeks?|days?)\b)',r'\1 ',text,flags=re.I)
    return text


def review_quantities(source,prediction):
    issues=[]
    original,output=dates(source),dates(prediction)
    for d in original:
        if d['ambiguous'] and d['raw'] not in prediction:
            issues.append({'code':'ambiguous_date','detail':'日期月日顺序不明确，请原样保留：'+d['raw']})
    # Do not reinterpret an ambiguous source date. For unambiguous dates, allow
    # faithful Chinese formatting instead of misclassifying the year as duration.
    clear=[d['canonical'] for d in original if not d['ambiguous']]
    clear_out=[d['canonical'] for d in output if not d['ambiguous']]
    if clear!=clear_out:
        issues.append({'code':'date_changed','detail':'日期与原文不一致或有遗漏，请核对并保留原日期'})
    if any(d['ambiguous'] and d['raw'] not in source for d in output):
        issues.append({'code':'ambiguous_date','detail':'输出新增了原文没有的日期'})
    source_years=[year for _,_,year in calendar_years(source)]
    output_years=[year for _,_,year in calendar_years(prediction)]
    if source_years!=output_years:
        issues.append({'code':'date_changed','detail':'日历年份与原文不一致或有遗漏，请保留原年份及其含义'})
    q=numeric_review(normalized_quantities_text(source),normalized_quantities_text(prediction))
    if q['warnings']:
        issues.append({'code':'quantity','detail':'；'.join(q['warnings'])})
    return q,issues
