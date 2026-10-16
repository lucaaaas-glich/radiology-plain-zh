from radplain.recheck_v2 import recheck
from radplain.predict_v2 import public_result


def sample(source, text, supported=True, valid=True, status='source_only', finish='stop'):
    return {'version':'old', 'use_critic':True, 'source_report':source, 'seconds':1.0,
        'glossary':[], 'segments':[{'id':1,'source':source,'start':0,'end':len(source),
        'status':status,'explanation_zh':text if status=='generated' else None,
        'attempts':[{'text':text,'finish_reason':finish,'critique':{'verdict':{
        'valid':valid,'supported':supported,'reason_zh':'old decision'}}}]}]}


def test_recheck_recovers_valid_spaced_age_without_mutating_history():
    original = sample('50-year-old woman.', '50 岁女性。')
    result = recheck(original)
    assert result['coverage']['fully_generated']
    assert original['segments'][0]['status']=='source_only'
    assert result['segments'][0]['attempts'][0]['critique']==original['segments'][0]['attempts'][0]['critique']
    assert result['postprocessing']['new_model_calls']==0


def test_recheck_revokes_old_approval_of_wrong_size_boundary():
    result = recheck(sample('Subcentimeter nodules.', '亚毫米结节。', status='generated'))
    assert result['segments'][0]['status']=='source_only'
    assert result['segments'][0]['explanation_zh'] is None
    assert 'attempts' not in public_result(result)['segments'][0]


def test_recheck_cannot_replace_negative_or_invalid_verdict_with_rule_pass():
    for supported,valid,finish in [(False,True,'stop'),(True,False,'stop'),(True,True,'length')]:
        result = recheck(sample('No pleural effusion.', '未见胸腔积液。', supported, valid, finish=finish))
        assert not result['coverage']['fully_generated']


def test_recheck_rejects_invented_advice_even_if_old_model_approved():
    result = recheck(sample('A nodule is indeterminate.', '结节性质待定，建议复查。',status='generated'))
    assert result['segments'][0]['status']=='source_only'


def test_recheck_keeps_device_identity_even_when_old_critic_approved_bowel_word():
    source = 'The enteric tube tip is below the diaphragm.'
    assert recheck(sample(source, '肠管尖端位于膈肌下方。', status='generated'))['segments'][0]['status']=='source_only'
    for valid in ('胃管尖端在膈肌下方。', '胃肠管尖端在膈肌下方。', '肠内营养管尖端在膈肌下方。'):
        assert recheck(sample(source, valid))['coverage']['fully_generated']
