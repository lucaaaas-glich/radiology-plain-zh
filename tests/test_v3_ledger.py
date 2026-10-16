import json
import pytest
from radplain.structure_v3 import units, chunks
from radplain.ledger_v3 import validate_ledger, validate_translations, validate_audit, parse_json
from radplain.explain_v3 import explain


def item(text='No significant right pleural effusion.'):
    return units(text)[0]


def ledger(text):
    return {'units':[{'id':1,'role':'finding','facts':[{'evidence':text,'entity':'pleural effusion','negation':'No significant','location':'right'}]}]}


def reply(value):
    return {'text':json.dumps(value,ensure_ascii=False),'finish_reason':'stop','seconds':.01,'input_tokens':10,'generation_tokens':10}


def test_quotes_cannot_invent_more_specific_anatomy():
    text='No significant right pleural effusion.'
    value=ledger(text)
    validate_ledger(value,[item(text)])
    value['units'][0]['facts'][0]['location']='right lower lobe'
    with pytest.raises(ValueError,match='grounded'):
        validate_ledger(value,[item(text)])


def test_evidence_must_not_drop_negative_qualifier():
    text='No significant right pleural effusion.'
    value=ledger(text)
    value['units'][0]['facts']=[{'evidence':'right pleural effusion'}]
    with pytest.raises(ValueError,match='uncovered'):
        validate_ledger(value,[item(text)])


def test_evidence_from_another_sentence_is_rejected():
    value=ledger('No significant right pleural effusion.')
    with pytest.raises(ValueError,match='exact source quote'):
        validate_ledger(value,[item('No pneumothorax.')])


def test_lossless_structure_preserves_labels_decimals_and_all_text():
    source='Findings\nI. A 3.5 mm right upper lung nodule.\nII. No significant adenopathy.\nImpression: stable.'
    rows=units(source)
    assert rows[0]['structure_only']
    assert rows[1]['source'].startswith('I. A 3.5 mm')
    assert rows[2]['source'].startswith('II. No significant')
    assert rows[-1]['section_hint']=='impression'
    assert all(source[r['start']:r['end']]==r['source'] for r in rows)
    assert ''.join(c for r in rows for c in r['source'] if not c.isspace())==''.join(c for c in source if not c.isspace())


def test_chunk_budget_does_not_truncate_long_single_unit():
    rows=units('x'*1400+'\nNo effusion.')
    groups=list(chunks(rows,max_chars=100))
    assert len(groups)==2 and len(groups[0][0]['source'])==1400


def test_report_list_label_after_colon_stays_with_its_content():
    source='Other findings: a. Pleural space: No findings. b. Heart enlarged.'
    rows=units(source)
    assert rows[0]['source'].strip()=='Other findings:'
    assert rows[1]['source'].startswith('a. Pleural space:')
    assert rows[2]['source'].startswith('b. Heart')
    assert ''.join(c for r in rows for c in r['source'] if not c.isspace())==''.join(c for c in source if not c.isspace())


def test_known_heading_omission_from_ledger_quote_is_not_clinical_omission():
    text='IV. Conclusion: No significant right pleural effusion.'
    value=ledger('No significant right pleural effusion.')
    validate_ledger(value,[item(text)])
    value['units'][0]['facts']=[{'evidence':'right pleural effusion.'}]
    with pytest.raises(ValueError,match='uncovered'):validate_ledger(value,[item(text)])


def test_projection_direction_guard_has_narrow_imaging_scope():
    from radplain.guard_v3 import review
    assert any(x['code']=='projection_direction' for x in review('Portable PA chest X-ray.','便携式前后位胸片。')['issues'])
    assert not any(x['code']=='projection_direction' for x in review('Portable PA chest X-ray.','便携式后前位胸片。')['issues'])
    assert not any(x['code']=='projection_direction' for x in review('PA diameter is normal.','前后位。')['issues'])


@pytest.mark.parametrize('value',[
    {'translations':[]},
    {'translations':[{'id':1,'text':'中文'},{'id':1,'text':'重复'}]},
    {'translations':[{'id':True,'text':'中文'}]},
    {'translations':[{'id':1,'text':''}]},
])
def test_missing_duplicate_or_invalid_outputs_are_not_accepted(value):
    with pytest.raises(ValueError):
        validate_translations(value,[item()])


def test_auditor_must_check_every_id_and_cannot_reference_foreign_source():
    with pytest.raises(ValueError):
        validate_audit({'checked_ids':[],'issues':[]},[item()])
    with pytest.raises(ValueError):
        validate_audit({'checked_ids':[1],'issues':[{'id':2,'type':'meaning','reason':'错'}]},[item()])


def test_truncated_json_is_never_accepted_even_if_parseable():
    with pytest.raises(ValueError):
        parse_json({'text':'{}','finish_reason':'length'})


def test_duplicate_json_fields_cannot_silently_erase_anatomy():
    with pytest.raises(ValueError,match='Duplicate JSON'):
        parse_json({'text':'{"facts":[{"location":"right subclavian vein","location":"superior vena cava"}]}','finish_reason':'stop'})


def test_numeric_omission_is_not_overridden_by_model_approval():
    text='A 5 mm nodule.'
    def generate(messages,max_tokens):
        system=messages[0]['content']
        if '你提取' in system:
            return reply({'units':[{'id':1,'role':'finding','facts':[{'evidence':text}]}]})
        if '你把目标' in system:
            return reply({'translations':[{'id':1,'text':'有一个结节。'}]})
        return reply({'checked_ids':[1],'issues':[]})
    result=explain(text,generate)
    assert not result['coverage']['fully_generated']
    assert len(result['calls'])==5
    assert result['segments'][0]['explanation_zh'] is None
    assert result['segments'][0]['source']==text


def test_bidirectional_omission_verdict_causes_bounded_repair():
    text='No significant right pleural effusion.'
    calls={'write':0}
    def generate(messages,max_tokens):
        system=messages[0]['content']
        if '你提取' in system:
            return reply(ledger(text))
        if '你把目标' in system:
            calls['write']+=1
            return reply({'translations':[{'id':1,'text':'未见'+('明显' if calls['write']==2 else '')+'右侧胸腔积液。'}]})
        return reply({'checked_ids':[1],'issues':[] if calls['write']==2 else [{'id':1,'type':'omission','reason':'遗漏significant限定'}]})
    result=explain(text,generate)
    assert result['coverage']['fully_generated']
    assert '明显' in result['segments'][0]['explanation_zh']
    assert len(result['segments'][0]['attempts'])==2


def test_schema_failure_keeps_source_and_does_not_leak_failed_candidate():
    result=explain('No pneumothorax.',lambda *a,**k:reply({'translations':[]}))
    assert len(result['calls'])==3
    assert result['segments'][0]['status']=='source_only'
    assert result['segments'][0]['explanation_zh'] is None


def test_no_ledger_ablation_performs_real_write_and_audit_without_extraction():
    def generate(messages,max_tokens):
        assert '你提取' not in messages[0]['content']
        if '你把目标' in messages[0]['content']:
            return reply({'translations':[{'id':1,'text':'未见气胸。'}]})
        return reply({'checked_ids':[1],'issues':[]})
    result=explain('No pneumothorax.',generate,use_ledger=False)
    assert result['coverage']['fully_generated'] and not result['use_ledger']
    assert len(result['calls'])==2 and result['segments'][0]['ledger'] is None


def test_bad_ledger_does_not_block_source_based_translation_or_appear_as_evidence():
    def generate(messages,max_tokens):
        system=messages[0]['content']
        if '你提取' in system:
            return reply({'units':[{'id':1,'role':'finding','facts':[{'evidence':'INVENTED FACT'}]}]})
        if '你把目标' in system:
            assert 'INVENTED FACT' not in messages[1]['content']
            return reply({'translations':[{'id':1,'text':'未见气胸。'}]})
        return reply({'checked_ids':[1],'issues':[]})
    result=explain('No pneumothorax.',generate)
    assert result['coverage']['fully_generated']
    assert result['coverage']['generated_from_source_without_ledger']==1
    assert result['segments'][0]['ledger'] is None
    assert result['segments'][0]['ledger_status']=='format_failed'


def test_critic_rejected_ledger_requires_new_translation_and_source_audit():
    writes=0
    def generate(messages,max_tokens):
        nonlocal writes
        system=messages[0]['content']
        if '你提取' in system:return reply({'units':[{'id':1,'role':'finding','facts':[{'evidence':'No pneumothorax.'}]}]})
        if '你把目标' in system:
            writes+=1
            return reply({'translations':[{'id':1,'text':'未见气胸。'}]})
        return reply({'checked_ids':[1],'issues':[{'id':1,'type':'ledger','reason':'清单归类不妥'}] if writes==1 else []})
    result=explain('No pneumothorax.',generate)
    assert writes==2 and result['coverage']['fully_generated']
    assert result['segments'][0]['ledger'] is None
    assert result['segments'][0]['ledger_status']=='critic_rejected'
