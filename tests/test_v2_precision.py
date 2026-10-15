from radplain.guard_v2 import review
from radplain.explain_v2 import critique_accepts


def codes(source, output):
    return {i['code'] for i in review(source, output)['issues']}


def test_lobe_and_projected_lung_zone_are_not_interchangeable():
    assert 'lower_lobe' in codes('Left lower lobe opacity.', '左下肺野有阴影。')
    assert 'upper_lobe' in codes('Nodules in both upper lobes.', '双肺上部有结节。')
    assert 'middle_lobe' in codes('Right middle lobe nodule.', '右肺中部有结节。')
    assert 'zone_became_lobe' in codes('Opacity at the left lung base.', '左肺下叶有阴影。')
    assert not codes('Left lower lobe opacity.', '左肺下叶有阴影。')
    assert not codes('Opacity at the left lung base.', '左肺底部有阴影。')


def test_critic_cannot_waive_lost_device_or_recommendation_subject():
    audit={'verdict':{'valid':True,'supported':True}}
    checks=[review('The endotracheal tube tip is 3 cm above the carina.', '在气管分叉上方3厘米。'),
            review('Follow-up imaging is recommended.', '进行影像复查。')]
    assert all(not critique_accepts(c,audit) for c in checks)


def test_glossary_maps_the_actual_english_term_without_asserting_a_finding():
    from radplain.glossary_v2 import glossary
    entries=glossary('No sigmoid colon lesion or hydronephrosis.')
    assert any(e['source_term']=='sigmoid colon' and e['term']=='乙状结肠' for e in entries)
    assert any(e['source_term']=='hydronephrosis' and e['term']=='肾积水' for e in entries)
    assert all('不表示报告肯定存在' in e['note'] for e in entries)
