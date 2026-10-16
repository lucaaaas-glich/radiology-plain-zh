"""V3 additions motivated by OLD development errors; V2 glossary unchanged."""
import re
from .glossary_v2 import glossary as old_glossary


def glossary(source):
    entries=old_glossary(source)
    match=re.search(r'\brespiratory\s+failure\b|呼吸衰竭',source,re.I)
    if match:
        entries.append({'term':'呼吸衰竭','source_term':match.group(),
            'definition_zh':'呼吸系统不能充分完成氧气进入血液或二氧化碳排出的气体交换；不能只用“呼吸困难”这个症状替代它。',
            'url':'https://www.nhlbi.nih.gov/health/respiratory-failure',
            'note':'词语释义，不表示报告肯定存在该发现。'})
    for code,meaning,direction in [('PA','后前位','从身体后方向前方投照'),('AP','前后位','从身体前方向后方投照')]:
        match=re.search(r'\b'+code+r'\s+(?:(?:portable|chest)\s+)*(?:X[- ]?ray|radiograph|view|projection)\b',source,re.I)
        if match:
            entries.append({'term':code+'胸片投照方向','source_term':match.group(),
                'definition_zh':code+'在这里指'+meaning+'，即X线'+direction+'；不要把PA和AP对调。',
                'url':'https://www.gov.uk/government/publications/diagnostic-radiology-national-diagnostic-reference-levels-ndrls/ndrl',
                'note':'只在明确胸片/投照语境中解释缩写；不猜测其他语境中的PA/AP。'})
    return entries
