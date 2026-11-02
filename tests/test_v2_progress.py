from radplain.explain_v2 import explain
from radplain.glossary_v2 import glossary


def test_progress_reports_completion_without_leaking_failed_candidate():
    calls=[]
    def generate(messages,max_tokens):
        return {'text':'private invalid candidate', 'seconds':0.01,
                'generation_tokens':4, 'finish_reason':'stop'}
    result=explain('No pneumothorax.',generate,progress=lambda *args:calls.append(args))
    assert calls==[(1,1,'source_only')]
    assert result['coverage']['fully_generated'] is False
    assert all('private' not in str(value) for value in calls)


def test_progress_is_observational_and_does_not_change_generated_content():
    def generate(messages,max_tokens):
        return {'text':'未见气胸。','seconds':0.01,'generation_tokens':5,'finish_reason':'stop'}
    calls=[]
    plain=explain('No pneumothorax.',generate)
    observed=explain('No pneumothorax.',generate,progress=lambda *args:calls.append(args))
    assert plain['segments']==observed['segments']
    assert plain['coverage']==observed['coverage']
    assert calls==[(1,1,'generated')]


def test_breast_reshaping_term_does_not_claim_removal_or_infer_surgery():
    for term in ('mammoplasty', 'mammaplasty'):
        entries=glossary('No history of '+term+'.')
        item=next(x for x in entries if x['term']=='乳房成形术')
        assert item['source_term']==term
        assert '切除' not in item['definition_zh']
        assert item['url']=='https://www.ncbi.nlm.nih.gov/mesh/68016462'
        assert item['note']=='词语释义，不表示报告肯定存在该发现。'


def test_sigmoid_definition_does_not_claim_broader_compound_equivalence():
    from radplain.glossary_v2 import glossary
    assert any(e['term'] == '乙状结肠' for e in glossary('sigmoid colon'))
    assert not any(e['term'] == '乙状结肠' for e in glossary('Colosigmoid diverticulosis'))
