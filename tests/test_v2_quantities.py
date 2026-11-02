from radplain.quantity_v2 import review_quantities


def test_spelled_months_and_half_months_match_chinese():
    for s,p in [('one month','一个月'),('three and a half months','3个半月'),('twenty-one days','二十一天'),('half a year','半年')]:
        assert not review_quantities(s,p)[1]


def test_dates_do_not_become_duration_years():
    assert not review_quantities('Compared with 19 06 2012.','与2012年6月19日相比。')[1]
    assert not review_quantities('Compared with 31/7/2020.','与2020年7月31日相比。')[1]


def test_ambiguous_date_cannot_be_guessed_even_if_plausible():
    assert review_quantities('03/07/2025','2025年3月7日')[1][0]['code']=='ambiguous_date'
    assert not review_quantities('03/07/2025','日期03/07/2025')[1]


def test_wrong_dates_and_wrong_durations_still_flag():
    assert review_quantities('2025-11-06','2025年11月7日')[1]
    assert review_quantities('three and a half months','3个月')[1]


def test_calendar_year_is_not_a_duration():
    assert not review_quantities('Surgery in 2015.', '2015年做过手术。')[1]
    assert not review_quantities('Stable since 2015.', '自2015年以来稳定。')[1]
    assert not review_quantities('2015年做过手术。', '手术在2015年进行。')[1]
    for output in ['2016年做过手术。', '做过手术。', '持续2015年。']:
        assert any(i['code']=='date_changed' for i in review_quantities('Surgery in 2015.', output)[1])


def test_year_like_number_cannot_be_reinterpreted_as_date():
    assert review_quantities('Report ID 2015.', '报告日期是2015年。')[1]
    assert not review_quantities('Follow up for 2015 years.', '随访持续2015年。')[1]
    assert review_quantities('Follow up for 2015 years.', '随访持续2016年。')[1]
    assert review_quantities('Observed in 2015 mm.', '观察于2015年。')[1]


def test_calendar_year_and_full_date_are_both_preserved():
    s='Surgery in 2015. Compared with 31/7/2020.'
    assert not review_quantities(s, '2015年做过手术。与2020年7月31日相比。')[1]
    assert review_quantities(s, '2016年做过手术。与2020年7月31日相比。')[1]


def test_added_dates_are_not_ignored():
    assert review_quantities('No findings.', '2020年7月31日未见异常。')[1]
    assert review_quantities('No findings.', '03/07/2020未见异常。')[1]


def test_spelled_hyphenated_age_keeps_its_quantity():
    assert not review_quantities('A three-year-old child.', '3岁儿童。')[1]
    assert not review_quantities('A 50-year-old woman.', '50岁女性。')[1]
    assert review_quantities('A three-year-old child.', '4岁儿童。')[1]
