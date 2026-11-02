"""Paired case transitions; errors and completion are counted separately."""
import argparse
import json
from pathlib import Path
from scripts.summarize_v2 import load_rows, validate, fully_generated


def compare(rows_a, review_a, rows_b, review_b):
    a = validate(rows_a, review_a)
    b = validate(rows_b, review_b)
    if set(a) != set(b):
        raise ValueError('Paired comparison requires identical case IDs')
    output_a = {r['id']: r['result'] for r in rows_a}
    output_b = {r['id']: r['result'] for r in rows_b}
    counts = {key: {'both': 0, 'only_a': 0, 'only_b': 0, 'neither': 0}
              for key in ('observed_error', 'complete_faithful', 'fully_generated')}
    cases = []
    for key in a:
        metrics = {
            'observed_error': (bool(a[key]['errors']), bool(b[key]['errors'])),
            'complete_faithful': (a[key]['complete_faithful_chinese'], b[key]['complete_faithful_chinese']),
            'fully_generated': (fully_generated(output_a[key]), fully_generated(output_b[key])),
        }
        case = {'id': key}
        for metric, (first, second) in metrics.items():
            state = ('both' if second else 'only_a') if first else ('only_b' if second else 'neither')
            counts[metric][state] += 1
            case[metric] = state
        cases.append(case)
    return {'paired_reports': len(a), 'counts': counts, 'cases': cases,
            'review_scope': 'project AI; not blinded, independent, or clinician review',
            'not_clinical_accuracy': True}


def main():
    p = argparse.ArgumentParser()
    for arm in ('a', 'b'):
        p.add_argument('--'+arm+'-outputs', required=True)
        p.add_argument('--'+arm+'-review', required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--stage', required=True, choices=['development', 'final'])
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite a paired comparison')
    if args.stage == 'final':
        barrier = json.loads(Path('examples/v2/final_inspection_barrier.json').read_text())
        if barrier['state'] != 'released':
            raise ValueError('Final inspection barrier is still held')
    result = compare(load_rows(args.a_outputs), json.loads(Path(args.a_review).read_text()),
                     load_rows(args.b_outputs), json.loads(Path(args.b_review).read_text()))
    result['stage'] = args.stage
    result['inputs'] = {k: v for k, v in vars(args).items() if k.endswith(('_outputs', '_review'))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
