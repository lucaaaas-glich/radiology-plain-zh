from radplain.quantity_v2 import review_quantities
from radplain.guard_v2 import review


def codes(s,p):return {i['code'] for i in review(s,p)['issues']}


def test_spaced_chinese_dates_keep_same_unambiguous_day():
    assert not review_quantities('Compared with 19 06 2012.','与 2012 年 06 月 19 日相比。')[1]
    assert review_quantities('Compared with 19 06 2012.','与 2012 年 06 月 20 日相比。')[1]
    assert review_quantities('03/07/2025','2025 年 3 月 7 日')[1]


def test_spaced_calendar_year_is_not_a_duration():
    assert not review_quantities('Surgery in 2015.','2015 年做过手术。')[1]
    assert review_quantities('Surgery in 2015.','2016 年做过手术。')[1]
    assert review_quantities('Surgery in 2015.','持续 2015 年。')[1]


def test_spaced_age_cannot_change_or_disappear():
    assert not review_quantities('A 50-year-old woman.','50 岁女性。')[1]
    assert not review_quantities('A three-year-old child.','3 周岁儿童。')[1]
    assert review_quantities('A 50-year-old woman.','51 岁女性。')[1]
    assert review_quantities('A 50-year-old woman.','女性。')[1]


def test_negated_signs_do_not_require_new_uncertainty_word():
    assert 'possible' not in codes('No pleural effusion or signs suggestive of pneumothorax are noted.','未见胸腔积液或气胸征象。')
    assert 'possible' not in codes('No signs suggestive of pneumothorax.','未见气胸征象。')
    assert 'possible' in codes('Signs suggestive of pneumothorax.','气胸征象。')


def test_positive_uncertainty_in_second_clause_stays_required():
    assert 'possible' in codes('No signs suggestive of pneumothorax; possible pneumonia.','未见气胸征象；肺炎。')


def test_spaced_half_duration_preserves_value_not_just_digits():
    from radplain.quantity_v2 import review_quantities
    for value in ('3 个半月', '三 个 半 月', '3个半月'):
        _, issues = review_quantities('three and a half months ago', value+'前')
        assert not issues
    for wrong in ('3 个月', '半月', '3 个半周'):
        _, issues = review_quantities('three and a half months ago', wrong+'前')
        assert issues
