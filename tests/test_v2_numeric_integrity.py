from radplain.guard_v2 import review
from radplain.explain_v2 import critique_accepts


def missing(source, text):
    return any(x['code']=='numeric_value_missing' for x in review(source, text)['issues'])


def test_missing_series_identifier_cannot_be_waived_by_critic():
    check=review('27 mm lesion, image 13 of series 5.', '病变27毫米，位于图像13。')
    assert missing('27 mm lesion, image 13 of series 5.', '病变27毫米，位于图像13。')
    assert not critique_accepts(check, {'verdict': {'valid':True,'supported':True}})


def test_reordering_does_not_equal_missing_number():
    assert not missing('27 mm lesion, image 13 of series 5.', '系列5的图像13显示27毫米病变。')


def test_normalized_dates_durations_and_covid_do_not_cause_missing_flags():
    pairs=[('Compared with 31/7/2020.', '与2020年7月31日比较。'),
           ('Follow up in three and a half months.', '3个半月后随访。'),
           ('No COVID-19 pneumonia.', '未见新冠肺炎。'),
           ('Surgery in 2015.', '2015年手术。'),
           ('A 50-year-old woman.', '50岁女性。')]
    for source,text in pairs:
        assert not missing(source,text)


def test_repeated_number_and_ordinal_cannot_disappear():
    assert missing('A 5 mm and another 5 mm nodule.', '发现5毫米结节。')
    assert missing('The sixth rib.', '第七肋骨。')
