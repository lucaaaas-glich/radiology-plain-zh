"""Reconcile original and continuation traces without selecting/retrying rows.

Use a local, SHA-verified copy of the remote run directory. All inputs and
request settings are checked against the frozen lock. Raw data stay ignored.
"""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.resume_v3_final import completed_rows


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def within(root, name):
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    return path


def merge_unique(destination, incoming):
    if set(destination) & set(incoming):
        raise ValueError('Duplicate completed case across allocations; never select a preferred result')
    destination.update(incoming)


def collect(root, lock_path):
    if digest(root / 'final_candidate.lock.json') != digest(lock_path):
        raise ValueError('Downloaded lock differs from the committed freeze')
    lock = json.loads(lock_path.read_text())
    originals, expected, merged, provenance = {}, {}, {}, {}
    file_hashes = {}

    def checked_json(name, sha=None):
        path = within(root, name)
        value = digest(path)
        if sha and value != sha:
            raise ValueError('Changed frozen file: ' + name)
        file_hashes[name] = value
        return json.loads(path.read_text())

    def append_trace(name, arm, allowed):
        path = within(root, name)
        if not path.exists():
            return
        incoming = completed_rows(path, allowed)
        merge_unique(merged[arm], incoming)
        file_hashes[name] = digest(path)
        for key in incoming:
            provenance[arm][key] = name

    for name, field in [('10-final20.json', 'main_request_sha256'),
                        ('11-ablation10.json', 'ablation_request_sha256')]:
        req = checked_json('requests/' + name, lock[field])
        source = within(root, req['input'])
        if digest(source) != req['input_sha256']:
            raise ValueError('Changed frozen source input')
        file_hashes[req['input']] = digest(source)
        rows = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
        indexed = {r['id']: r for r in rows}
        if len(indexed) != len(rows):
            raise ValueError('Duplicate source ID')
        originals[name] = req
        for arm in req['arms']:
            if arm in expected:
                raise ValueError('Repeated arm definition')
            expected[arm] = indexed
            merged[arm], provenance[arm] = {}, {}
            append_trace(req['out'] + '/' + arm + '/outputs.jsonl', arm, indexed)

    for plan_path in sorted((root / 'results').glob('resume-*/plan.json')):
        plan = checked_json(str(plan_path.relative_to(root)))
        for part in plan['parts']:
            req = checked_json(part['request'], part['request_sha256'])
            original = originals[part['original_request']]
            if set(req['arms']) != set(part['arms']):
                raise ValueError('Continuation arm mismatch')
            source = within(root, req['input'])
            if digest(source) != req['input_sha256']:
                raise ValueError('Continuation input changed')
            file_hashes[req['input']] = digest(source)
            rows = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
            if [r['id'] for r in rows] != part['ids'] or len(set(part['ids'])) != len(rows):
                raise ValueError('Continuation cases changed')
            for arm, config in req['arms'].items():
                if config != original['arms'][arm]:
                    raise ValueError('Continuation changed the frozen generation settings')
                if any(row != expected[arm].get(row['id']) for row in rows):
                    raise ValueError('Continuation changed the source report')
                append_trace(req['out'] + '/' + arm + '/outputs.jsonl', arm,
                             {r['id']: r for r in rows})
    missing = {arm: [key for key in expected[arm] if key not in merged[arm]]
               for arm in expected}
    ordered = {arm: [merged[arm][key] for key in expected[arm] if key in merged[arm]]
               for arm in expected}
    return ordered, {'complete': not any(missing.values()), 'missing': missing,
                     'counts': {arm: len(rows) for arm, rows in ordered.items()},
                     'source_files_sha256': file_hashes, 'row_provenance': provenance,
                     'candidate_lock_sha256': digest(lock_path),
                     'policy': 'Keep every completed result, including failures; reject duplicates; only unfinished final JSON lines may be ignored.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--lock', type=Path, default=Path('examples/v3/final_candidate.lock.json'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--allow-incomplete', action='store_true',
                   help='Explicit time-limit closeout only; never a complete final evaluation')
    a = p.parse_args()
    rows, manifest = collect(a.root.resolve(), a.lock)
    if not manifest['complete'] and not a.allow_incomplete:
        raise ValueError('Incomplete evaluation: ' + json.dumps(manifest['counts']))
    a.out.mkdir(parents=True, exist_ok=False)
    for arm, values in rows.items():
        path = a.out / arm / 'outputs.jsonl'
        path.parent.mkdir()
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values))
        manifest.setdefault('merged_files_sha256', {})[arm] = digest(path)
    (a.out / 'collection_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'complete': manifest['complete'], 'counts': manifest['counts']}))


if __name__ == '__main__':
    main()
