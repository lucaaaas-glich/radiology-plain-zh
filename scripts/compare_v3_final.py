"""Compare complete frozen cohorts, separating corrections from withholding."""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.summarize_v2 import load_rows
from scripts.summarize_v3 import compare, faithful, validate, verify_cohort


def paired_analysis(rows_a, review_a, rows_b, review_b):
    result = compare(rows_a, review_a, rows_b, review_b)
    a, b = validate(rows_a, review_a), validate(rows_b, review_b)
    result['old_error_to_complete_faithful'] = [
        key for key in a if a[key]['errors'] and faithful(b[key])]
    result['old_error_to_incomplete_without_observed_error'] = [
        key for key in a if a[key]['errors'] and not b[key]['errors']
        and not b[key]['clinical_coverage_complete']]
    result['old_error_to_complete_but_precision_concern'] = [
        key for key in a if a[key]['errors'] and not b[key]['errors']
        and b[key]['clinical_coverage_complete'] and b[key]['precision_concern']]
    result['complete_faithful_gained'] = [
        key for key in a if not faithful(a[key]) and faithful(b[key])]
    result['complete_faithful_lost'] = [
        key for key in a if faithful(a[key]) and not faithful(b[key])]
    ta = sum(row['generation_accounting']['seconds'] for row in rows_a)
    tb = sum(row['generation_accounting']['seconds'] for row in rows_b)
    result['paired_generation_seconds'] = {'a': ta, 'b': tb,
                                           'b_over_a': tb / ta if ta else None}
    result['interpretation'] = (
        'Error disappearance with incomplete output is withholding, not demonstrated '
        'correction. All clinical judgments are project AI review, not independent '
        'medical accuracy. A complete-faithful label means no error was observed.')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--v2-outputs', type=Path, required=True)
    p.add_argument('--v3-outputs', type=Path, required=True)
    p.add_argument('--without-ledger-outputs', type=Path, required=True)
    p.add_argument('--review-directory', type=Path, default=Path('results/v3/final'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite paired evaluation')
    manifest_path = Path('examples/v3/main20_manifest.json')
    lock_path = Path('examples/v3/final_candidate.lock.json')
    manifest = json.loads(manifest_path.read_text())
    lock_path.read_bytes()  # Require the pre-output candidate freeze.
    files = {'v2': args.v2_outputs, 'v3': args.v3_outputs,
             'without_ledger': args.without_ledger_outputs}
    rows = {arm: load_rows(path) for arm, path in files.items()}
    reviews = {}
    for arm in files:
        path = args.review_directory / (arm + '_review.json')
        reviews[arm] = json.loads(path.read_text())
        verify_cohort(rows[arm], manifest,
                      'ablation' if arm == 'without_ledger' else 'final')
        validate(rows[arm], reviews[arm])
    control_ids = {row['id'] for row in manifest['rows'][::2]}
    with_rows = [row for row in rows['v3'] if row['id'] in control_ids]
    with_review = dict(reviews['v3'], cases=[
        case for case in reviews['v3']['cases'] if case['id'] in control_ids])
    result = {
        'main_v2_a_vs_v3_b': paired_analysis(
            rows['v2'], reviews['v2'], rows['v3'], reviews['v3']),
        'control_without_ledger_a_vs_with_ledger_b': paired_analysis(
            rows['without_ledger'], reviews['without_ledger'], with_rows, with_review),
        'not_clinical_accuracy': True,
        'sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in [*files.values(), manifest_path, lock_path,
                                *[args.review_directory / (arm + '_review.json')
                                  for arm in files]]},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
