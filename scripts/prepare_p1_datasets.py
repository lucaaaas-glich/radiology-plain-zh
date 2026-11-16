"""Prepare provisional matched data and a small GPU pilot; no inference or labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
from pathlib import Path

from radplain.common import fingerprint, read_jsonl, write_json, write_jsonl


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_output_path(path, root, buckets, *, check_tmp=False):
    """Reject history paths, symlinks and prior outputs before any writes."""
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    allowed = [root / bucket for bucket in buckets]
    resolved = path.resolve()
    bucket = next((value for value in allowed
                   if resolved != value and resolved.is_relative_to(value)), None)
    if bucket is None:
        raise ValueError('Output must remain under a permitted new P0/P1 location')
    if bucket.resolve() != bucket or any(part.is_symlink() for part in (path, *path.parents)
                                         if part.is_relative_to(root)):
        raise ValueError('Output paths must not contain symlinks')
    candidates = [path]
    if check_tmp:
        candidates.append(path.with_suffix(path.suffix + '.tmp'))
    if any(candidate.exists() or candidate.is_symlink() for candidate in candidates):
        raise FileExistsError('Refusing to overwrite an output or temporary file')
    return path


def verify_quality_inputs(summary_path, train_path, risk_path):
    """A report hash alone cannot bind lexical flags to the screened target."""
    summary = json.loads(summary_path.read_text())
    if summary['input']['sha256'] != digest(train_path):
        raise ValueError('Training input changed since the frozen quality screening')
    if summary['local_artifacts']['per_sample.jsonl']['sha256'] != digest(risk_path):
        raise ValueError('Per-sample risks changed since the quality summary')


def representative(rows, groups, seed):
    rows = sorted(rows, key=lambda row: row['id'])
    random.Random(seed).shuffle(rows)
    chosen = {}
    for row in rows:
        chosen.setdefault(groups[row['id']], row)
    return list(chosen.values())


def stratum(row, risk, config):
    features = dict(risk['features'])
    features['target_length_bin'] = sum(len(row['reference_zh']) >= boundary
                                         for boundary in config['target_character_bin_boundaries'])
    return tuple(features[key] for key in config['matching_features'])


def allocate(raw_counts, clean_counts, size):
    """Capped largest-remainder allocation; both policies use identical strata."""
    keys = sorted(raw_counts)
    if size > sum(clean_counts.get(key, 0) for key in keys):
        raise ValueError('Insufficient no-known-flag capacity')
    total = sum(raw_counts.values())
    ideal = {key: size * raw_counts[key] / total for key in keys}
    result = {key: min(int(ideal[key]), clean_counts.get(key, 0)) for key in keys}
    while sum(result.values()) < size:
        available = [key for key in keys if result[key] < clean_counts.get(key, 0)]
        key = max(available, key=lambda k: (ideal[k] - result[k], raw_counts[k], k))
        result[key] += 1
    return result


def choose_arms(rows, risks, config):
    raw, clean = defaultdict(list), defaultdict(list)
    for row in rows:
        key = stratum(row, risks[row['id']], config)
        raw[key].append(row)
        if risks[row['id']]['screen_status'] == 'no_known_rule_flag':
            clean[key].append(row)
    size = min(config['requested_per_arm'], sum(map(len, clean.values())))
    if size < config['minimum_per_arm']:
        raise ValueError('Not enough eligible distinct groups for a formal-size candidate set')
    counts = allocate({k: len(v) for k, v in raw.items()}, {k: len(v) for k, v in clean.items()}, size)
    arms = {'S0': [], 'S1': []}
    for offset, key in enumerate(sorted(raw)):
        candidates = sorted(raw[key], key=lambda row: row['id'])
        random.Random(config['seed'] + 1000 + offset).shuffle(candidates)
        # A shared random ordering makes the selection policy the only difference.
        arms['S0'].extend(candidates[:counts[key]])
        arms['S1'].extend([r for r in candidates if risks[r['id']]['screen_status'] == 'no_known_rule_flag'][:counts[key]])
    for name in arms:
        random.Random(config['seed'] + 2000).shuffle(arms[name])
    return arms, counts


def common_support_summary(rows, risks, config):
    raw = Counter(stratum(row, risks[row['id']], config) for row in rows)
    clean = Counter(stratum(row, risks[row['id']], config) for row in rows
                    if risks[row['id']]['screen_status'] == 'no_known_rule_flag')
    unsupported = {key: count for key, count in raw.items() if not clean[key]}
    return {'raw_strata': len(raw), 'strata_without_no_known_flag_candidates': len(unsupported),
            'raw_reports_in_unsupported_strata': sum(unsupported.values()),
            's0_scope': 'Common-support stratified random sample, capped by S1 capacity; not an unconditional random sample of all eligible reports',
            'unsupported_strata': [{'features': list(key), 'raw_reports': count}
                                   for key, count in sorted(unsupported.items())]}


def prepare(root, config_path, out, summary_path, quality_summary_path=None):
    root = root.resolve()
    out = new_output_path(out, root, ('data/p0',))
    summary_path = new_output_path(summary_path, root, ('results/p0', 'results/p1'), check_tmp=True)
    config = json.loads(config_path.read_text())
    input_paths = {'train': root / 'data/pilot_v1/train.zh.jsonl',
                   'dev': root / 'data/pilot_v1/validation.zh.jsonl',
                   'groups': root / 'data/p0/pre-gpu-20261116/grouping/groups.json',
                   'risks': root / 'data/p0/pre-gpu-20261116/quality/per_sample.jsonl',
                   'quality_summary': root / (quality_summary_path or
                                              Path('results/p0/pre-gpu-20261116/quality_summary.json'))}
    verify_quality_inputs(input_paths['quality_summary'], input_paths['train'], input_paths['risks'])
    groups = json.loads(input_paths['groups'].read_text())['groups']
    all_train, all_dev = read_jsonl(input_paths['train']), read_jsonl(input_paths['dev'])
    risk_rows = read_jsonl(input_paths['risks'])
    risks = {r['id']: r for r in risk_rows}
    if len(risks) != len(risk_rows) or set(risks) != {r['id'] for r in all_train}:
        raise ValueError('Risk records do not exactly cover original training candidates')
    blocked_sources = ['data/pilot_v1/validation.jsonl', 'data/pilot_v1/test.jsonl',
                       'data/v2/real/dev.jsonl', 'data/v2/real/holdout.jsonl',
                       'data/v3/real/holdout.jsonl', 'examples/challenge.jsonl',
                       'examples/v2/holdout.jsonl']
    blocked_groups = set()
    for source in blocked_sources:
        path = root / source
        input_paths['exclude:' + source] = path
        blocked_groups.update(groups[fingerprint(r['report'])] for r in read_jsonl(path))
    excluded, eligible = [], []
    for row in all_train:
        key = fingerprint(row['report'])
        risk = risks[row['id']]
        if row['id'] != key or risk['source_sha256'] != hashlib.sha256(row['report'].encode()).hexdigest():
            raise ValueError('Source identity mismatch')
        if risk.get('normalized_source_sha256', key) != key:
            raise ValueError('Normalized risk source mismatch')
        if risk['teacher_accepted'] != row['teacher_accepted']:
            raise ValueError('Format acceptance changed')
        reason = ('teacher_format_rejected' if not row['teacher_accepted'] else
                  'text_group_shared_with_prior_heldout_or_development' if groups[key] in blocked_groups else None)
        if reason:
            excluded.append({'id': row['id'], 'reason': reason})
        else:
            eligible.append(row)
    unique_train = representative(eligible, groups, config['seed'])
    arms, strata = choose_arms(unique_train, risks, config)
    dev_pool = representative([r for r in all_dev if r['teacher_accepted']], groups, config['seed'] + 1)
    if len(dev_pool) < config['development_reports']:
        raise ValueError('Insufficient grouped development data')
    dev = dev_pool[:config['development_reports']]
    if any(groups[r['id']] in {groups[d['id']] for d in dev} for rows in arms.values() for r in rows):
        raise ValueError('Training/development text-group collision')
    out.mkdir(parents=True, exist_ok=False)
    for arm, rows in arms.items():
        write_jsonl(out / arm / 'train.zh.jsonl', rows)
        write_jsonl(out / arm / 'validation.zh.jsonl', dev)
    pilot = arms['S0'][:config['pilot_train_size']]
    write_jsonl(out / 'pilot/train.zh.jsonl', pilot)
    write_jsonl(out / 'pilot/validation.zh.jsonl', dev[:config['pilot_validation_size']])
    write_json(out / 'excluded_training.json', excluded)
    stats = {}
    for arm, rows in arms.items():
        stats[arm] = {'reports': len(rows), 'distinct_text_groups': len({groups[r['id']] for r in rows}),
                      'flagged_reports': sum(bool(risks[r['id']]['flags']) for r in rows),
                      'source_characters': sum(len(r['report']) for r in rows),
                      'target_characters': sum(len(r['reference_zh']) for r in rows),
                      'features': {name: dict(Counter(str(risks[r['id']]['features'][name]) for r in rows))
                                   for name in config['matching_features'] if name != 'target_length_bin'}}
    equal = (Counter(stratum(r, risks[r['id']], config) for r in arms['S0']) ==
             Counter(stratum(r, risks[r['id']], config) for r in arms['S1']))
    if not equal:
        raise ValueError('Stratum balance failed')
    summary = {'version': config['version'], 'status': config['status'],
               'config_sha256': digest(config_path), 'inputs': {k: {'path': str(p.relative_to(root)), 'sha256': digest(p)}
                                                               for k, p in input_paths.items()},
               'original_training': len(all_train), 'exclusions': dict(Counter(r['reason'] for r in excluded)),
               'eligible_before_group_representatives': len(eligible), 'distinct_group_representatives': len(unique_train),
               'group_redundant_candidates_not_selected': len(eligible) - len(unique_train),
               'arms': stats, 'identical_joint_stratum_counts': equal,
               'common_support': common_support_summary(unique_train, risks, config),
               'matched_strata': [{'features': list(key), 'per_arm': n} for key, n in sorted(strata.items()) if n],
               'shared_training_reports_between_arms': len({r['id'] for r in arms['S0']} & {r['id'] for r in arms['S1']}),
               'development_reports': len(dev), 'pilot': {'train': len(pilot), 'validation': config['pilot_validation_size']},
               'human_quality_reviews_completed': 0, 'new_final_set_selected': False,
               'formal_training_quality_gate_passed': False,
               'interpretation': 'Provisional risk-screening policy comparison. No-known-flag is not clinical correctness; matched features do not remove all confounding.',
               'local_output_files': {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*.jsonl'))},
               'ids': {arm: [r['id'] for r in rows] for arm, rows in arms.items()},
               'dev_ids': [r['id'] for r in dev]}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--config', type=Path, default=Path('configs/p1_data_selection.json'))
    p.add_argument('--out', type=Path, default=Path('data/p0/pre-gpu-20261116/prepared'))
    p.add_argument('--summary', type=Path, default=Path('results/p0/pre-gpu-20261116/data_selection.json'))
    p.add_argument('--quality-summary', type=Path,
                   help='Frozen screen summary binding the training input and per_sample hashes; defaults to results/p0/pre-gpu-20261116/quality_summary.json')
    a = p.parse_args()
    result = prepare(a.root.resolve(), a.config, a.out, a.summary, a.quality_summary)
    print(json.dumps({k: result[k] for k in ('original_training', 'exclusions', 'distinct_group_representatives',
                                           'identical_joint_stratum_counts', 'development_reports', 'pilot')}, indent=2))
