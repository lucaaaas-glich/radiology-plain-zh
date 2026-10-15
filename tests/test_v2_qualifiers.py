import pytest
from radplain.guard_v2 import review
from radplain.explain_v2 import critique_accepts
from radplain.glossary_v2 import glossary


@pytest.mark.parametrize('source,good,bad,code',[
    ('Mild cardiomegaly.', '心脏轻度增大。','心脏增大。','degree_mild'),
    ('Moderate pleural effusion.', '中等量胸腔积液。','胸腔积液。','degree_moderate'),
    ('Marked emphysema.', '明显肺气肿。','肺气肿。','degree_marked'),
    ('Subtle ground-glass opacity.', '细微磨玻璃影。','磨玻璃影。','degree_subtle'),
    ('Multiple renal cysts.', '多个肾囊肿。','肾囊肿。','multiple'),
])
def test_degree_and_count_loss_cannot_be_waived(source,good,bad,code):
    assert code not in {x['code'] for x in review(source,good)['issues']}
    check=review(source,bad)
    assert code in {x['code'] for x in check['issues']}
    assert not critique_accepts(check,{'verdict':{'valid':True,'supported':True}})


def test_negative_mention_is_definition_only_not_a_patient_finding():
    items=glossary('No lymphadenopathy or metastasis. Post right hepatectomy.')
    assert {x['term'] for x in items}=={'淋巴结肿大／病变','转移','肝切除术'}
    assert all(x['note']=='词语释义，不表示报告肯定存在该发现。' for x in items)


@pytest.mark.parametrize('source,text,code',[
    ('Scarring in both lung apices.','双肺上叶有瘢痕。','lung_apex'),
    ('Fibrosis remains in the lingula.','左中叶舌段仍有纤维化。','left_middle_lobe'),
    ('Follow-up.','本次检查建议复查。','followup_context_became_advice'),
])
def test_observed_development_scope_errors_fail_closed(source,text,code):
    check=review(source,text)
    assert code in {x['code'] for x in check['issues']}
    assert not critique_accepts(check,{'verdict':{'valid':True,'supported':True}})


def test_correct_apex_and_followup_are_not_blocked():
    assert not review('Scarring in both lung apices.','双肺尖有瘢痕。')['issues']
    assert not review('Follow-up.','随访。')['issues']
