import copy
import pytest
from scripts.summarize_v2 import fingerprint, summarize, template, validate


def fixture_rows():
    return [
        {'id': 'fallback', 'result': {'seconds': 1, 'coverage': {'fully_generated': False},
                                    'segments': [{'status': 'source_only', 'attempts': [{}, {}]}]}},
        {'id': 'wrong', 'result': {'seconds': 2, 'text': '错误解释', 'finish_reason': 'stop'}},
        {'id': 'good', 'result': {'seconds': 3, 'text': '完整解释', 'finish_reason': 'stop'}},
    ]


def reviewed(rows):
    r = template(rows)
    for c in r['cases']:
        c.update(errors=[], complete_faithful_chinese=False, readability='mixed', notes_zh='')
    r['cases'][1].update(errors=['number_time'], notes_zh='数值对应的器官错了。')
    r['cases'][2].update(complete_faithful_chinese=True, readability='plain')
    return r


def test_fallback_is_neither_error_free_success_nor_complete_translation():
    rows = fixture_rows()
    result = summarize(rows, reviewed(rows))
    assert result['reports_with_observed_factual_errors'] == 1
    assert result['reports_complete_faithful_chinese'] == 1
    assert result['reports_fully_generated'] == 2
    assert result['reports_no_observed_error_but_incomplete'] == 1
    assert result['evidence_segments']['source_only'] == 1


def test_missing_or_stale_reviews_cannot_be_scored():
    rows = fixture_rows()
    with pytest.raises(ValueError):
        summarize(rows, template(rows))
    review = reviewed(rows)
    changed = copy.deepcopy(rows)
    changed[1]['result']['text'] = '后来修改的输出'
    with pytest.raises(ValueError, match='changed output'):
        validate(changed, review)
    review['cases'].pop()
    with pytest.raises(ValueError, match='exactly one'):
        validate(rows, review)


def test_cannot_mark_fallback_as_faithful_success():
    rows = fixture_rows()
    review = reviewed(rows)
    review['cases'][0]['complete_faithful_chinese'] = True
    with pytest.raises(ValueError, match='incomplete'):
        validate(rows, review)
