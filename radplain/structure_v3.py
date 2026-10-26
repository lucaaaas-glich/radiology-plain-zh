"""Lossless, conservative report units; labels are never clinical conclusions."""
import re
from .guard_v2 import spans

HEADINGS = {
    'clinical information': 'history', 'clinical history': 'history',
    'history': 'history', 'indication': 'history', 'indications': 'history',
    'reason for request': 'history', 'clinical indication': 'history',
    'technique': 'technique', 'procedure': 'technique', 'examination': 'technique',
    'findings': 'finding', 'results': 'finding',
    'impression': 'impression', 'conclusion': 'impression', 'conclusions': 'impression',
    'recommendation': 'recommendation', 'recommendations': 'recommendation',
}
LABEL = re.compile(r'^(?:[IVXLCDM]{1,6}|\d{1,2}|[A-Za-z])\s*[.):]?$', re.I)
PREFIX = re.compile(r'^(?:[IVXLCDM]{1,6}|\d{1,2}|[A-Za-z])[.)]\s+',re.I)


def structural_prefix_length(text):
    """Only explicit list labels and known headings, never clinical words."""
    prefix=PREFIX.match(text)
    offset=prefix.end() if prefix else 0
    tail=text[offset:]
    if ':' in tail and tail.split(':',1)[0].strip().casefold() in HEADINGS:
        offset+=tail.index(':')+1
    return offset


def units(report):
    if not report.strip():
        raise ValueError('Empty report')
    if len(report) > 12000:
        raise ValueError('Report exceeds 12000 characters; no truncation')
    raw = spans(report)
    for index in range(len(raw)-1):
        item=raw[index]
        match=re.search(r':\s+((?:\d{1,2}|[A-Za-z])\.)$',item['source'])
        if match:
            boundary=item['start']+match.start(1)
            item['end']=boundary
            item['source']=report[item['start']:boundary]
            raw[index+1]['start']=boundary
            raw[index+1]['source']=report[boundary:raw[index+1]['end']]
    merged = []
    i = 0
    while i < len(raw):
        item = dict(raw[i])
        # A standalone enumeration belongs to the following text. Preserve the
        # intervening whitespace and exact offsets rather than inventing content.
        if LABEL.fullmatch(item['source']) and i + 1 < len(raw):
            i += 1
            item['end'] = raw[i]['end']
            item['source'] = report[item['start']:item['end']]
        merged.append(item)
        i += 1
    role = 'unknown'
    for n, item in enumerate(merged, 1):
        text = item['source'].strip()
        labelled=PREFIX.sub('',text)
        heading = labelled.split(':', 1)[0].strip().casefold().rstrip('.')
        pure = labelled.rstrip(':.').strip().casefold() in HEADINGS
        if heading in HEADINGS and (':' in text or pure):
            role = HEADINGS[heading]
        item.update(id=n, section_hint=role, structure_only=pure)
    # Every non-whitespace source character must remain visible somewhere.
    covered = set()
    for item in merged:
        assert report[item['start']:item['end']] == item['source']
        covered.update(range(item['start'], item['end']))
    if any(not c.isspace() and i not in covered for i, c in enumerate(report)):
        raise ValueError('Parser omitted source characters')
    return merged


def chunks(items, max_chars=1100, max_units=6):
    """Never split a sentence or silently truncate an oversized single unit."""
    group, size = [], 0
    for item in items:
        if item['structure_only']:
            continue
        if group and (size + len(item['source']) > max_chars or len(group) >= max_units):
            yield group
            group, size = [], 0
        group.append(item)
        size += len(item['source'])
    if group:
        yield group
