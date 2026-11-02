import pytest
from scripts.compare_v2_reviews import compare
from scripts.summarize_v2 import template


def fixture(complete, error, case_id='same'):
    rows = [{'id':case_id, 'result':{'text':'示例', 'seconds':1,
                                    'finish_reason':'stop' if complete else 'length'}}]
    review = template(rows)
    review['cases'][0].update(errors=['negation'] if error else [],
        complete_faithful_chinese=complete and not error, readability='mixed', notes_zh='复核示例')
    return rows, review


def test_error_disappearing_into_abstention_is_not_faithful_improvement():
    result = compare(*fixture(True, True), *fixture(False, False))
    assert result['counts']['observed_error']['only_a'] == 1
    assert result['counts']['fully_generated']['only_a'] == 1
    assert result['counts']['complete_faithful']['neither'] == 1


def test_different_cases_cannot_be_reported_as_paired():
    rows, review = fixture(True, False, 'different')
    with pytest.raises(ValueError, match='identical case IDs'):
        compare(*fixture(True, False), rows, review)
