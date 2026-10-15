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
DATES=re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}([/ -])\d{1,2}\1\d{4}\b|\d{4}年\d{1,2}月\d{1,2}日')


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
    text=DATES.sub(' [DATE] ',text)
    text=re.sub(r'COVID[ -]?19','COVID',text,flags=re.I)
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
    if clear and clear!=clear_out:
        issues.append({'code':'date_changed','detail':'日期与原文不一致或有遗漏，请核对并保留原日期'})
    q=numeric_review(normalized_quantities_text(source),normalized_quantities_text(prediction))
    if q['warnings']:
        issues.append({'code':'quantity','detail':'；'.join(q['warnings'])})
    return q,issues
