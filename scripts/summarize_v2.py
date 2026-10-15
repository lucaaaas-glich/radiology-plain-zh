"""Audit-friendly outcome counts. Never infer factual correctness from guards.

Prepare a blank review with --template, fill it case by case, then summarize.
Review files contain IDs/hashes and paraphrased notes, not source reports.
"""
import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

CATEGORIES = {
    'number_time', 'negation', 'certainty', 'anatomy_side', 'change',
    'recommendation', 'omission', 'unsupported_addition', 'terminology',
}
READABILITY = {'plain', 'mixed', 'technical', 'none'}


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty results or duplicate case IDs')
    return rows


def fingerprint(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def fully_generated(result):
    if 'coverage' in result:
        return result['coverage']['fully_generated']
    return result.get('finish_reason') == 'stop' and bool(result.get('text', '').strip())


def template(rows):
    return {
        'reviewer': 'project AI; not independent, blinded, or clinician review',
        'scope': 'Displayed Chinese only; source-only fallback is incomplete, not a correct answer.',
        'cases': [{'id': r['id'], 'output_sha256': fingerprint(r), 'errors': None,
                   'complete_faithful_chinese': None, 'readability': None,
                   'notes_zh': None} for r in rows],
    }


def validate(rows, review):
    cases = review['cases']
    if len(cases) != len(rows) or len({r['id'] for r in cases}) != len(cases):
        raise ValueError('Every output requires exactly one review')
    indexed = {r['id']: r for r in cases}
    if indexed.keys() != {r['id'] for r in rows}:
        raise ValueError('Review case IDs do not match output IDs')
    for row in rows:
        case = indexed[row['id']]
        if case['output_sha256'] != fingerprint(row):
            raise ValueError('Review references changed output: ' + row['id'])
        if not isinstance(case['errors'], list) or any(x not in CATEGORIES for x in case['errors']):
            raise ValueError('Missing or invalid error categories: ' + row['id'])
        if len(set(case['errors'])) != len(case['errors']):
            raise ValueError('Duplicate category: ' + row['id'])
        if type(case['complete_faithful_chinese']) is not bool:
            raise ValueError('Missing factual completeness judgment: ' + row['id'])
        if case['readability'] not in READABILITY or not isinstance(case['notes_zh'], str):
            raise ValueError('Missing readability/notes: ' + row['id'])
        if case['errors'] and not case['notes_zh'].strip():
            raise ValueError('Errors require a concrete explanation: ' + row['id'])
        if case['complete_faithful_chinese'] and (case['errors'] or not fully_generated(row['result'])):
            raise ValueError('Erroneous or incomplete output cannot be fully faithful Chinese: ' + row['id'])
    return indexed


def summarize(rows, review):
    indexed = validate(rows, review)
    durations = [r['result']['seconds'] for r in rows]
    categories = Counter(x for c in review['cases'] for x in c['errors'])
    segments = [s for r in rows for s in r['result'].get('segments', [])]
    eligible = [s for s in segments if s['status'] != 'excluded_instruction']
    result = {
        'review_scope': review['reviewer'], 'not_clinical_accuracy': True,
        'reports': len(rows),
        'reports_with_observed_factual_errors': sum(bool(c['errors']) for c in review['cases']),
        'reports_complete_faithful_chinese': sum(c['complete_faithful_chinese'] for c in review['cases']),
        'reports_fully_generated': sum(fully_generated(r['result']) for r in rows),
        'reports_no_observed_error_but_incomplete': sum(
            not indexed[r['id']]['errors'] and not fully_generated(r['result']) for r in rows),
        'error_category_report_counts': dict(sorted(categories.items())),
        'readability_ai_judgments': dict(Counter(c['readability'] for c in review['cases'])),
        'latency_seconds': {'total': sum(durations), 'median': statistics.median(durations),
                            'minimum': min(durations), 'maximum': max(durations)},
        'generation_completeness_note': 'Generation completion is mechanical, not a factual or clinical score.',
    }
    if segments:
        generated = sum(s['status'] == 'generated' for s in eligible)
        result['evidence_segments'] = {
            'eligible': len(eligible), 'generated': generated,
            'source_only': len(eligible) - generated,
            'coverage': generated / len(eligible) if eligible else None,
            'excluded_instruction': len(segments) - len(eligible),
            'repair_attempted': sum(len(s['attempts']) > 1 for s in eligible),
            'repair_accepted': sum(len(s['attempts']) > 1 and s['status'] == 'generated' for s in eligible),
        }
    return result


def paired_counts(left, right):
    a = {c['id']: bool(c['errors']) for c in left['cases']}
    b = {c['id']: bool(c['errors']) for c in right['cases']}
    if a.keys() != b.keys():
        raise ValueError('Paired reviews must contain the same cases')
    return {'both_have_error': sum(a[k] and b[k] for k in a),
            'left_only_error': sum(a[k] and not b[k] for k in a),
            'right_only_error': sum(not a[k] and b[k] for k in a),
            'neither_observed_error': sum(not a[k] and not b[k] for k in a),
            'note': 'No observed error includes fallback. Read coverage and faithful-completion counts alongside this.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outputs', required=True, type=Path)
    p.add_argument('--review', type=Path)
    p.add_argument('--template', action='store_true')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        p.error('Refusing to overwrite existing artifact')
    rows = load_rows(a.outputs)
    if a.template:
        result = template(rows)
    elif a.review:
        result = summarize(rows, json.loads(a.review.read_text()))
        result['outputs_file_sha256'] = hashlib.sha256(a.outputs.read_bytes()).hexdigest()
        result['review_file_sha256'] = hashlib.sha256(a.review.read_bytes()).hexdigest()
    else:
        p.error('Use --template or provide a completed --review')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
