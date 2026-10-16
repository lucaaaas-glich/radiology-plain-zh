"""Re-screen stored candidates using current guards and unchanged old verdicts.

No model call or new repair. This is a separately evaluated postprocessing arm,
not a claim that historical model verdicts were recomputed with new prompts.
"""
from copy import deepcopy
import time
from .guard_v2 import review, VERSION as GUARD_VERSION
from .explain_v2 import critique_accepts

VERSION = 'stored-candidate-recheck-v1'


def recheck(result):
    if 'postprocessing' in result:
        raise ValueError('Do not recheck an already postprocessed result')
    if result.get('use_critic') is not True:
        raise ValueError('Recheck requires original semantic-critic records')
    started = time.perf_counter()
    updated = deepcopy(result)
    updated.pop('display_text', None)
    changed = []
    for row in updated['segments']:
        before = (row['status'], row.get('explanation_zh'))
        row['original_status'] = row['status']
        if row['status'] == 'excluded_instruction':
            continue
        # Missing, truncated or invalid original verdicts cannot authorize text.
        chosen = None
        for index, attempt in enumerate(row['attempts']):
            check = review(row['source'], attempt['text'], attempt.get('finish_reason'))
            verdict = attempt.get('critique', {}).get('verdict', {})
            valid_record = (type(verdict.get('valid')) is bool
                            and type(verdict.get('supported')) is bool)
            accepted = valid_record and critique_accepts(check, {'verdict': verdict})
            attempt['recheck'] = {'version': VERSION, 'guard_version': GUARD_VERSION,
                                  'check': check, 'accepted': accepted,
                                  'critic_verdict_reused': True}
            if accepted and chosen is None:
                chosen = attempt['text']
                row['selected_original_attempt_index'] = index
        if chosen is not None:
            row.update(status='generated', explanation_zh=chosen)
            row.pop('reason', None)
        else:
            row.update(status='source_only', explanation_zh=None,
                       reason='已有候选仍有需核对的信息，保留原句，未完成此句的中文解释。')
        if before != (row['status'], row.get('explanation_zh')):
            changed.append(row['id'])
    eligible = [row for row in updated['segments'] if row['status'] != 'excluded_instruction']
    generated = sum(row['status'] == 'generated' for row in eligible)
    updated['coverage'] = {'generated_segments': generated, 'eligible_segments': len(eligible),
                          'source_only_segments': len(eligible)-generated,
                          'excluded_instruction_segments': len(updated['segments'])-len(eligible),
                          'fully_generated': bool(eligible) and generated == len(eligible)}
    elapsed = time.perf_counter()-started
    updated['postprocessing'] = {'version': VERSION, 'guard_version': GUARD_VERSION,
        'source_version': result['version'], 'changed_segment_ids': changed,
        'seconds': elapsed, 'new_model_calls': 0, 'new_repairs': 0,
        'original_generation_seconds': result['seconds'],
        'note': 'Original semantic verdicts and glossary reused, not newly generated or clinically validated.'}
    updated['version'] = result['version']+'+'+VERSION
    updated['seconds'] = result['seconds']+elapsed
    return updated
