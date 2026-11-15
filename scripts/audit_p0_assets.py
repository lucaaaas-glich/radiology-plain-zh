"""Read-only, model-free V1 recovery checks; export metadata, never report text."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from radplain.common import fingerprint, read_jsonl, write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(rows):
    result = {r['id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate IDs')
    return result


def audit(root):
    summary, used_ids = {}, set()
    for split in ('train', 'validation', 'test'):
        paths = [root / 'data/pilot_v1' / (split + suffix)
                 for suffix in ('.jsonl', '.zh.jsonl', '.teacher-journal.jsonl')]
        raw, zh, journal = [unique(read_jsonl(path)) for path in paths]
        if set(raw) != set(zh) or zh != journal:
            raise ValueError('Raw/teacher record mismatch: ' + split)
        if any(any(zh[key].get(k) != v for k, v in row.items()) for key, row in raw.items()):
            raise ValueError('Teacher changed a source field: ' + split)
        if any(fingerprint(row['report']) != key for key, row in raw.items()):
            raise ValueError('Source fingerprint mismatch: ' + split)
        if used_ids.intersection(raw):
            raise ValueError('Exact V1 source overlap')
        used_ids.update(raw)
        if any(type(row.get('teacher_accepted')) is not bool for row in zh.values()):
            raise ValueError('Missing explicit format acceptance: ' + split)
        summary[split] = {'rows': len(raw), 'format_accepted': sum(r['teacher_accepted'] for r in zh.values()),
                          'source_counts': dict(Counter(r['source'] for r in raw.values())),
                          'raw_fields_preserved': True, 'journal_equal_by_id': True,
                          'files': {str(p.relative_to(root)): digest(p) for p in paths}}
    cfg = json.loads((root / 'configs/pilot.json').read_text())
    training = json.loads((root / 'artifacts/pilot_v1/training.json').read_text())
    selection = json.loads((root / 'artifacts/pilot_v1/selection.json').read_text())
    if training['config'] != cfg:
        raise ValueError('Training configuration mismatch')
    for split in ('train', 'validation'):
        if training['filters'][split]['used'] != summary[split]['format_accepted']:
            raise ValueError('Historical training count differs; investigate filtering')
    adapter_hashes = {}
    for name in ('best', 'epoch-1', 'epoch-2'):
        path = root / 'artifacts/pilot_v1' / name / 'adapter_model.safetensors'
        adapter_hashes[name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    if adapter_hashes['best']['sha256'] != adapter_hashes[f"epoch-{selection['epoch']}"]['sha256']:
        raise ValueError('Best adapter does not match selected epoch')
    locked = json.loads((root / 'results/pilot_v1/models.lock.json').read_text())
    weights = {}
    for name in ('base-model', 'merged-model', 'mlx-4bit', 'v2/qwen35-4b'):
        path = root / 'artifacts' / name
        files = list(path.glob('*.safetensors')) if path.exists() else []
        weights[name] = {'directory_present': path.is_dir(), 'weight_files': len(files),
                         'weight_bytes': sum(p.stat().st_size for p in files),
                         'full_weight_hash_rechecked': False, 'model_load_tested': False}
    return {'phase': 'P0_initial_local_audit', 'audit_time_utc': datetime.now(timezone.utc).isoformat(),
            'data': summary, 'v1_exact_cross_split_overlap': 0, 'adapters': adapter_hashes,
            'model_versions': {k: {f: v[f] for f in ('model', 'revision')} for k, v in locked.items()},
            'weight_inventory': weights,
            'historical_training': {k: training[k] for k in
                                    ('trainable_parameters', 'optimizer_steps', 'elapsed_seconds', 'gpu')},
            'checks': {'jsonl_parse_and_alignment': True, 'adapter_sha_recomputed': True,
                       'clinical_fidelity_reviewed': False, 'near_duplicates_reviewed': False,
                       'new_training_started': False, 'gpu_connected': False},
            'limitations': ['Format acceptance is not factual correctness.',
                            'New local hashes are snapshots, not proof of equality to an absent historic hash.',
                            'Only small adapter weights were fully hashed; larger models were not loaded.',
                            'Merged and MLX V1 weights include old fine-tuning and cannot be B0.',
                            'V1 saves adapters, not optimizer/scheduler/RNG/data-progress resume states.',
                            'Dataset usage terms need source verification; source availability is not permission.',
                            'This audit does not claim a new independent final evaluation set exists.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Refusing to overwrite an audit')
    value = audit(args.root.resolve())
    write_json(args.out, value)
    print(json.dumps({'rows': {k: v['rows'] for k, v in value['data'].items()},
                      'format_accepted_train': value['data']['train']['format_accepted']}, indent=2))


if __name__ == '__main__':
    main()
