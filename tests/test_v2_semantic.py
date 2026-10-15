from radplain.semantic_v2 import parse_verdict
from radplain.explain_v2 import explain,render_text,critique_accepts


def test_only_explicit_complete_boolean_verdict_can_pass():
    for text,finish in [('yes','stop'),('{"supported":"true","reason_zh":"ok"}','stop'),
                         ('{"supported":true,"reason_zh":"ok"}','length'),
                         ('{"supported":true}','stop')]:
        assert not parse_verdict(text,finish)['supported']


def test_reasoning_is_not_returned_as_verdict():
    r=parse_verdict('<think>private intermediate text</think>\n{"supported":false,"reason_zh":"左右写反"}','stop')
    assert r=={'supported':False,'reason_zh':'左右写反','valid':True}


def test_valid_json_code_fence_is_accepted_without_making_clinical_claim():
    r=parse_verdict('```json\n{"supported":true,"reason_zh":"未发现差异"}\n```','stop')
    assert r['supported'] and r['valid']


def test_model_approval_cannot_overrule_wrong_quantity_or_lost_scope():
    audit={'verdict':{'supported':True,'valid':True}}
    for code in ['quantity','ambiguous_date','date_changed','newness','acute_scope','lung_base']:
        assert not critique_accepts({'issues':[{'code':code}]},audit)


def test_model_critique_can_resolve_soft_numeric_order_flag():
    assert critique_accepts({'issues':[{'code':'numeric_sequence'}]}, {'verdict':{'supported':True,'valid':True}})


def test_semantic_repair_still_has_only_two_candidates_and_hides_failed_text():
    responses=iter([
        '右侧积液减少，左侧积液增多。',
        '{"supported":false,"reason_zh":"两侧变化写反，应为右增多左减少"}',
        '右侧积液增多，左侧积液减少。',
        '{"supported":true,"reason_zh":"两侧变化与原文一致"}',
    ])
    calls=[]
    def generate(messages,**kwargs):
        calls.append(messages)
        return {'text':next(responses),'finish_reason':'stop'}
    r=explain('Right effusion increased, left effusion decreased.',generate,use_critic=True)
    assert len(calls)==4
    assert r['coverage']['fully_generated']
    assert '右侧积液减少，左侧积液增多' not in render_text(r)


def test_line_protocol_keeps_embedded_quotes_and_fails_on_conflicting_verdicts():
    assert parse_verdict('SUPPORTED\n原文"Heart size"与心脏大小对应。','stop')['supported']
    assert not parse_verdict('UNSUPPORTED\n左右写反了。','stop')['supported']
    assert not parse_verdict('SUPPORTED','stop')['valid']
    assert not parse_verdict('SUPPORTED\nUNSUPPORTED','stop')['valid']
    assert not parse_verdict('SUPPORTED\n一致。','length')['valid']
