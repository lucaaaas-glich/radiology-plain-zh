import pytest
from radplain.guard_v2 import review
from radplain.explain_v2 import critique_accepts
from radplain.quantity_v2 import size_bounds


@pytest.mark.parametrize('source',[
    'Subcentimeter nodules.', 'Subcentimetric nodules.', 'Sub-centimetre nodules.', '亚厘米结节。',
])
def test_lexical_size_limit_can_be_explained_with_a_number(source):
    assert not review(source,'小于1厘米的结节。')['issues']
    assert not review(source,'不足一厘米的结节。')['issues']
    assert not review(source,'亚厘米结节。')['issues']


@pytest.mark.parametrize('bad',[
    '亚毫米结节。', '1厘米的结节。', '不超过1厘米的结节。',
    '大于1厘米的结节。', '小于2厘米的结节。', '不小于1厘米的结节。',
])
def test_wrong_scale_or_inequality_cannot_be_waived(bad):
    check=review('Subcentimeter nodules.',bad)
    assert any(x['code']=='quantity' for x in check['issues'])
    assert not critique_accepts(check,{'verdict':{'valid':True,'supported':True}})


def test_negated_comparison_is_not_interpreted_as_less_than():
    assert size_bounds('No less than 1 cm.')==[('>=','1','cm')]
    assert size_bounds('不小于1厘米。')==[('>=','1','cm')]
    assert not review('Nodule no less than 1 cm.','结节不小于1厘米。')['issues']
    assert any(x['code']=='quantity' for x in review('Nodule no less than 1 cm.','结节小于1厘米。')['issues'])


def test_do_not_add_an_inequality_to_an_exact_measurement():
    assert any(x['code']=='quantity' for x in review('A 1 cm nodule.','小于1厘米的结节。')['issues'])
    assert size_bounds('At most 2 cm.')==size_bounds('不超过2厘米。')


@pytest.mark.parametrize('source',['Probable infection.', 'This is probably an infection.'])
def test_probable_remains_uncertain(source):
    good=review(source,'可能是感染。')
    assert 'possible' not in {x['code'] for x in good['issues']}
    bad=review(source,'这是感染。')
    assert 'possible' in {x['code'] for x in bad['issues']}
    assert not critique_accepts(bad,{'verdict':{'valid':True,'supported':True}})
