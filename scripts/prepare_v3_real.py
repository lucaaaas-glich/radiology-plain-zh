"""Freeze new long public reports without reading any model predictions."""
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from radplain.common import fingerprint, read_jsonl, write_json, write_jsonl


def shingles(text):
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[i:i+5]) for i in range(max(0, len(words)-4))}


def near(a, b):
    return bool(a and b) and len(a & b) / len(a | b) >= .8


def main():
    dest = Path('data/v3/real')
    if dest.exists():
        raise ValueError('Frozen data already exists; refusing to replace it')
    previous = [p for folder in ('data/pilot_v1', 'data/v2/real')
                for p in Path(folder).glob('*.jsonl')]
    old = {fingerprint(r['report']): r['report'] for p in previous
           for r in read_jsonl(p) if 'report' in r}
    old_sets = [shingles(t) for t in old.values() if len(t) >= 700]
    source = Path('data/v2/source/data/validation-00000-of-00001.parquet')
    rows = pq.read_table(source).to_pylist()
    pool = {}
    for row in rows:
        report = str(row.get('radiology_report') or '').strip()
        key = fingerprint(report)
        if 1000 <= len(report) <= 4000 and key not in old:
            pool[key] = {'id': 'long_' + key[:16], 'report': report,
                         'source_sha256': key, 'length_chars': len(report),
                         'dataset_source': row['source']}
    rng = random.Random(37102603)
    candidates = sorted(pool.values(), key=lambda r: r['id'])
    rng.shuffle(candidates)
    selected, rejected_near = [], []
    strata = [[] for _ in range(3)]
    seen_sets = list(old_sets)
    for row in candidates:
        tokens = shingles(row['report'])
        if any(near(tokens, x) for x in seen_sets):
            rejected_near.append(row['id'])
            continue
        seen_sets.append(tokens)
        group = 0 if row['length_chars'] < 1500 else (1 if row['length_chars'] < 2000 else 2)
        row['length_stratum'] = group
        strata[group].append(row)
    for group, count in zip(strata, (28, 8, 4)):
        if len(group) < count:
            raise ValueError('Insufficient predeclared stratum: ' + str(len(group)))
        selected.extend(group[:count])
    rng.shuffle(selected)
    dest.mkdir(parents=True)
    write_jsonl(dest/'holdout.jsonl', selected)
    manifest = {
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'outputs_seen': False,
        'dataset': 'BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track',
        'revision': '6783ab38f9de28c4372799346f51d3927ae7eb9a', 'original_split': 'validation',
        'source_parquet_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'seed': 37102603, 'length_range_chars': [1000, 4000],
        'strata': ['1000–1499', '1500–1999', '2000–4000'], 'selected_counts': [28, 8, 4],
        'eligible_after_dedup_counts': list(map(len, strata)),
        'excluded_previous_unique_reports': len(old), 'near_duplicate_jaccard_threshold': .8,
        'near_duplicate_method': 'casefolded word 5-gram sets; old reports >=700 chars and all retained candidates',
        'rejected_near_ids': rejected_near, 'count': len(selected),
        'input_sha256': hashlib.sha256((dest/'holdout.jsonl').read_bytes()).hexdigest(),
        'rows': [{k: v for k, v in row.items() if k != 'report'} for row in selected],
        'limitations': 'Public English reports, not Chinese hospital originals; patient independence unknown. Long means relative to this dataset. No clinician Chinese reference. Raw text stays untracked.'}
    write_json('examples/v3/real_manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('count','eligible_after_dedup_counts','excluded_previous_unique_reports','input_sha256')}, indent=2))


if __name__ == '__main__':
    main()
