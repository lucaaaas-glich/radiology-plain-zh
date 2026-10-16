from scripts.summarize_v2_critic_diagnostic import summarize


def row(bad, valid, supported):
    return {'expected_problem': bad, 'result': {
        'verdict': {'valid': valid, 'supported': supported},
        'generation': {'seconds': 1, 'generation_tokens': 10,
                       'finish_reason': 'stop' if valid else 'reasoning_budget'}}}


def test_invalid_fail_closed_is_not_credited_as_detection():
    result = summarize([row(True, False, False), row(True, True, False),
                        row(False, False, False), row(False, True, True)])
    assert result['bad_pairs_validly_rejected'] == 1
    assert result['invalid_bad_pairs'] == 1
    assert result['invalid_valid_pairs'] == 1
    assert result['valid_pairs_accepted'] == 1
    assert result['valid_pairs_validly_rejected'] == 0
    assert result['finish_reasons']['reasoning_budget'] == 2


def test_valid_semantic_errors_remain_visible():
    result = summarize([row(True, True, True), row(False, True, False)])
    assert result['bad_pairs_accepted'] == 1
    assert result['valid_pairs_validly_rejected'] == 1
    assert result['bad_pairs_validly_rejected'] == 0
