"""Small, sourced glossary: definitions only, no patient-specific inference."""
import re

NCI = 'https://www.cancer.gov/publications/dictionaries/cancer-terms/def/'
ENTRIES = [
    ('胸腔积液', r'pleural\s+effusion|胸腔积液', '肺和胸壁之间的间隙里积聚了液体。', NCI + 'pleural-effusion'),
    ('气胸', r'pneumothorax|气胸', '肺和胸壁之间的间隙里进入了空气。', NCI + 'pneumothorax'),
    ('肺不张', r'atelectasis|肺不张', '部分肺组织没有充分展开。', NCI + 'atelectasis'),
    ('纵隔', r'mediastin\w*|纵隔', '两肺之间的区域，心脏、气管等结构位于其中。', NCI + 'mediastinum'),
    ('结节', r'\bnodules?\b|结节', '组织里的小团块；这个词本身不能确定它的性质。', NCI + 'nodule'),
    ('囊肿', r'\bcysts?\b|囊肿', '组织内形成的囊状结构，里面可能含有液体等物质。', NCI + 'cyst'),
    ('钙化', r'calcif\w*|钙化', '钙盐在组织中沉积。', NCI + 'calcification'),
    ('隆突／气管分叉', r'\bcarina\b|隆突|气管分叉', '气管分成左右主支气管的地方。', 'https://training.seer.cancer.gov/lung/anatomy/lung-anatomy-icdO3.html'),
]


def glossary(source):
    return [{'term': term, 'definition_zh': definition, 'url': url,
             'note': '词语释义，不表示报告肯定存在该发现。'}
            for term, pattern, definition, url in ENTRIES if re.search(pattern, source, re.I)]
