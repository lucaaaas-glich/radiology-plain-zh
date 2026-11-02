"""Summarize critic decisions without crediting malformed refusals as detection."""
import argparse
import json
from pathlib import Path


def summarize(records):
    result = {'pairs': len(records), 'bad_pairs': 0, 'valid_pairs': 0,
              'bad_pairs_validly_rejected': 0, 'bad_pairs_accepted': 0,
              'valid_pairs_accepted': 0, 'valid_pairs_validly_rejected': 0,
              'invalid_bad_pairs': 0, 'invalid_valid_pairs': 0,
              'seconds': 0, 'generated_tokens': 0, 'finish_reasons': {}}
    for row in records:
        bad = bool(row['expected_problem'])
        verdict = row['result']['verdict']
        generation = row['result']['generation']
        result['bad_pairs' if bad else 'valid_pairs'] += 1
        if not verdict['valid']:
            key = 'invalid_bad_pairs' if bad else 'invalid_valid_pairs'
        elif bad:
            key = 'bad_pairs_accepted' if verdict['supported'] else 'bad_pairs_validly_rejected'
        else:
            key = 'valid_pairs_accepted' if verdict['supported'] else 'valid_pairs_validly_rejected'
        result[key] += 1
        result['seconds'] += generation['seconds']
        result['generated_tokens'] += generation.get('generation_tokens', 0)
        reason = str(generation.get('finish_reason'))
        result['finish_reasons'][reason] = result['finish_reasons'].get(reason, 0) + 1
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('folder', type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    original = json.loads((a.folder/'summary.json').read_text())
    expected = original['plan']['count']
    arms = {}
    for arm in ('direct', 'thinking'):
        rows = [json.loads(s) for s in (a.folder/(arm+'.jsonl')).read_text().splitlines()]
        if len(rows) != expected:
            raise ValueError('Incomplete diagnostic; do not publish as complete')
        arms[arm] = summarize(rows)
    value = {'stage': 'development_diagnostic', 'not_clinical_accuracy': True,
             'invalid_is_not_correct_detection': True, 'plan': original['plan'],
             'arms': arms, 'source_files_sha256': original['source_files_sha256'],
             'source_changed_during_run': original['source_changed_during_run'],
             'load_seconds': original['load_seconds'],
             'peak_mlx_allocated_bytes': original['peak_mlx_allocated_bytes']}
    a.out.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
