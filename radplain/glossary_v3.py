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
    return entries
