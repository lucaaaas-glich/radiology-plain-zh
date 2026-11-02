import pytest
from radplain.guard_v2 import review
from radplain.explain_v2 import critique_accepts
from radplain.glossary_v2 import glossary


def codes(source, prediction):
    return {i['code'] for i in review(source, prediction)['issues']}


@pytest.mark.parametrize('source', [
    'Findings already known from a CT scan several years ago.',
    'Findings were previously known from imaging.',
    '这些发现已知。',
])
def test_historical_positive_findings_cannot_be_reversed(source):
    check = review(source, '此前检查未发现异常。')
    assert 'prior_findings_reversed' in {i['code'] for i in check['issues']}
    assert not critique_accepts(check, {'verdict': {'valid': True, 'supported': True}})


def test_history_rule_is_narrow_and_accepts_positive_paraphrase():
    source = 'Findings already known from a CT scan several years ago.'
    for target in ('这些发现几年前的CT检查就已显示。', '这些是此前已经知道的发现。'):
        assert 'prior_findings_reversed' not in codes(source, target)
    assert 'prior_findings_reversed' not in codes('No known previous findings.', '既往未发现异常。')
    assert 'prior_findings_reversed' not in codes('No new findings compared with previous CT.', '与此前CT相比未见新发现。')
    assert 'prior_findings_reversed' not in codes('Findings already known; no pleural effusion.', '这些发现此前已知；未见胸腔积液。')


def test_dorsal_spine_means_thoracic_region_not_added_posterior_location():
    source = 'Mild anterior marginal osteophytes in the dorsal spine.'
    assert 'thoracic_spine' not in codes(source, '胸椎前缘可见轻度骨赘。')
    check = review(source, '后侧脊柱前缘有轻度骨赘。')
    assert 'thoracic_spine' in {i['code'] for i in check['issues']}
    assert not critique_accepts(check, {'verdict': {'valid': True, 'supported': True}})
    assert not any(e['term'] == '胸椎／胸段脊柱' for e in glossary('dorsal root ganglion'))
    bad = review(source, '胸椎后侧可见轻度前缘骨赘。')
    assert 'dorsal_spine_added_posterior' in {i['code'] for i in bad['issues']}
    assert not critique_accepts(bad, {'verdict': {'valid': True, 'supported': True}})
    assert 'dorsal_spine_added_posterior' not in codes('Posterior osteophytes in the dorsal spine.', '胸椎后侧骨赘。')


@pytest.mark.parametrize(('english', 'target'), [('upper', '上'), ('lower', '下'), ('middle', '中')])
def test_valid_lung_lobe_word_order_does_not_force_fallback(english, target):
    source = f'A right {english} lobe nodule.'
    for translated in (f'右肺{target}叶结节。', f'右{target}肺叶结节。'):
        assert english+'_lobe' not in codes(source, translated)
    assert english+'_lobe' in codes(source, f'右肺{target}部结节。')
    assert 'left_middle_lobe' in codes('A left mid-zone nodule.', '左中肺叶结节。')
    assert 'zone_became_lobe' in codes('A nodule at the right lung base.', '右下肺叶结节。')


def test_model_approval_cannot_waive_invented_followup_advice():
    check = review('A right upper lobe nodule is indeterminate.', '右上肺叶结节性质待定，建议复查。')
    assert 'added_advice' in {i['code'] for i in check['issues']}
    assert not critique_accepts(check, {'verdict': {'valid': True, 'supported': True}})
    valid = review('Follow-up CT is recommended.', '报告建议复查CT。')
    assert 'added_advice' not in {i['code'] for i in valid['issues']}
