"""Small, sourced glossary: definitions only, no patient-specific inference."""
import re

NCI = 'https://www.cancer.gov/publications/dictionaries/cancer-terms/def/'
FLEISCHNER = 'https://doi.org/10.1148/radiol.232558'
ENTRIES = [
    ('胸腔积液', r'pleural\s+effusion|胸腔积液', '肺和胸壁之间的间隙里积聚了液体。', NCI + 'pleural-effusion'),
    ('气胸', r'pneumothorax|气胸', '肺和胸壁之间的间隙里进入了空气。', NCI + 'pneumothorax'),
    ('肺不张', r'atelectasis|肺不张', '部分肺组织没有充分展开。', NCI + 'atelectasis'),
    ('纵隔', r'mediastin\w*|纵隔', '两肺之间的区域，心脏、气管等结构位于其中。', NCI + 'mediastinum'),
    ('结节', r'\bnodules?\b|结节', '组织里的小团块；这个词本身不能确定它的性质。', NCI + 'nodule'),
    ('囊肿', r'\bcysts?\b|囊肿', '组织内形成的囊状结构，里面可能含有液体等物质。', NCI + 'cyst'),
    ('钙化', r'calcif\w*|钙化', '钙盐在组织中沉积。', NCI + 'calcification'),
    ('隆突／气管分叉', r'\bcarina\b|隆突|气管分叉', '气管分成左右主支气管的地方。', 'https://training.seer.cancer.gov/lung/anatomy/lung-anatomy-icdO3.html'),
    ('实变', r'\bconsolidation\b|实变', '部分肺内原本含空气的区域变得更密，在影像上更白；这个词本身不说明具体病因。', FLEISCHNER),
    ('磨玻璃影', r'ground.glass|磨玻璃', 'CT上出现较淡的发白区域，其中的血管和气道轮廓仍能辨认；这是影像外观描述。', FLEISCHNER),
    ('肺门', r'\bhil(?:um|a|ar)\b|肺门', '支气管、血管等结构进出肺的地方。', FLEISCHNER),
    ('心脏增大', r'\bcardiomegaly\b|心脏增大', '心脏的整体大小或某一部分比通常更大；这个词不确定具体原因。', 'https://my.clevelandclinic.org/health/diseases/21490-enlarged-heart-cardiomegaly'),
    ('肺气肿', 'emphysema|肺气肿', '肺里的小气囊异常膨胀，囊壁受到损伤。', 'https://www.cancer.gov/publications/dictionaries/cancer-terms/def/emphysema'),
    ('纤维化', 'fibro(?:sis|tic)|纤维化', '组织里的纤维成分增多。', 'https://www.cancer.gov/publications/dictionaries/cancer-terms/def/fibrosis'),
    ('乙状结肠', '(?:colo)?sigmoid(?:\\s+colon)?|乙状结肠', '结肠中连接直肠的S形一段。', 'https://www.cancer.gov/publications/dictionaries/cancer-terms/def/sigmoid-colon'),
    ('膈肌', 'diaphragm\\w*|膈肌', '分隔胸腔和腹腔的薄层肌肉。', 'https://www.cancer.gov/publications/dictionaries/cancer-terms/def/diaphragm'),
    ('支气管扩张', 'bronchiectasis|支气管扩张', '肺内输送空气的管道异常变宽，管壁受到损伤。', 'https://www.nhlbi.nih.gov/health/bronchiectasis'),
    ('肾积水', 'hydronephrosis|肾积水', '肾脏里收集尿液的肾盂扩大，也叫扩张。', 'https://www.niddk.nih.gov/health-information/urologic-diseases/hydronephrosis-newborns'),
    ('结肠憩室', 'diverticulosis|结肠憩室', '结肠壁向外形成小袋状突起；这个词本身不表示已经发炎。', 'https://www.niddk.nih.gov/health-information/digestive-diseases/diverticulosis-diverticulitis/definition-facts'),
    ('心包', 'pericardi\\w*|心包', '包在心脏周围的薄囊，其内层之间有少量液体。', 'https://www.cancer.gov/publications/dictionaries/cancer-terms/def/pericardium'),
    ('胸膜下', 'subpleural|胸膜下', '靠近覆盖肺表面的胸膜的位置描述；不是指肺泡下面。', 'https://doi.org/10.1148/radiol.232558'),
    ('肺叶', '\\blobe\\b|肺叶|[上下中]叶', '肺的主要分区。右肺有上、中、下三叶；左肺有上、下两叶。', 'https://doi.org/10.1148/radiol.232558'),
]


def glossary(source):
    return [{'term': term, 'source_term': re.search(pattern,source,re.I).group(0),
             'definition_zh': definition, 'url': url,
             'note': '词语释义，不表示报告肯定存在该发现。'}
            for term, pattern, definition, url in ENTRIES if re.search(pattern, source, re.I)]
