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
