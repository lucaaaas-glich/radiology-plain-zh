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


def test_chart_does_not_hide_error_by_counting_fallback_as_success():
    from scripts.plot_v2_outcomes import buckets
    rows=fixture_rows();review=reviewed(rows)
    review['cases'][0].update(errors=['negation'],notes_zh='一段回退，但另一段仍有否定错误。')
    result=buckets(rows,review)
    assert result=={'faithful_complete':1,'observed_error':2,
                   'incomplete_without_observed_error':0,'remaining_precision_concern':0}


def test_chart_keeps_unresolved_precision_concern_separate():
    from scripts.plot_v2_outcomes import buckets
    rows=fixture_rows();review=reviewed(rows)
    review['cases'][1].update(errors=[],notes_zh='未判明确错误，但不够精确。')
    result=buckets(rows,review)
    assert result=={'faithful_complete':1,'observed_error':0,
                   'incomplete_without_observed_error':1,'remaining_precision_concern':1}


def test_stored_candidate_recovery_is_not_counted_as_a_new_model_repair():
    rows = fixture_rows()
    rows[0]['result'].update(
        coverage={'fully_generated': True},
        postprocessing={'new_model_calls': 0, 'seconds': .01})
    segment = rows[0]['result']['segments'][0]
    segment.update(status='generated', original_status='source_only')
    result = summarize(rows, reviewed(rows))
    assert result['postprocessing']['restored_source_only_segments'] == 1
    assert result['postprocessing']['new_model_calls'] == 0
    assert 'repair_attempted' not in result['evidence_segments']
    assert result['evidence_segments']['original_trace_multiple_attempts'] == 1
    assert result['latency_seconds']['total'] == 6
