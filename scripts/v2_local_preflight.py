"""Validate the committed local lock and inspection release before final inference.

Pure Python: no model import, loading, downloads, or GPU side effects.
"""
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def committed(path):
    subprocess.run(['git', 'ls-files', '--error-unmatch', '--', str(path)],
                   check=True, capture_output=True)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', str(path)], check=True)


def validate(lock_path, arm_name, model, mode, max_tokens, input_path, split,
             barrier_path='examples/v2/final_inspection_barrier.json'):
    lock_path, barrier_path = Path(lock_path), Path(barrier_path)
    committed(lock_path)
    committed(barrier_path)
    lock = json.loads(lock_path.read_text())
    barrier = json.loads(barrier_path.read_text())
    if barrier['state'] != 'released' or not barrier.get('released_at_utc'):
        raise ValueError('Final inspection barrier is still held')
    if (barrier.get('local_candidate_lock') != str(lock_path)
            or barrier.get('local_candidate_lock_sha256') != digest(lock_path)):
        raise ValueError('Barrier is not bound to this local candidate lock')
    candidate_commit = barrier.get('local_candidate_commit')
    if not candidate_commit:
        raise ValueError('Barrier must name the local candidate commit')
    subprocess.run(['git', 'merge-base', '--is-ancestor', candidate_commit, 'HEAD'], check=True)
    saved = subprocess.check_output(['git', 'show', candidate_commit + ':' + str(lock_path)])
    if hashlib.sha256(saved).hexdigest() != digest(lock_path):
        raise ValueError('Candidate commit contains a different lock')
    arm = lock['arms'][arm_name]
    if (model, mode, max_tokens) != (arm['model'], arm['mode'], arm['max_tokens']):
        raise ValueError('Requested arm differs from committed candidate')
    key = 'real' if split == 'real-holdout' else 'synthetic'
    if digest(input_path) != lock['holdout_sha256'][key]:
        raise ValueError('Holdout differs from candidate lock')
    for path, sha in lock['code_sha256'].items():
        committed(path)
        if digest(path) != sha:
            raise ValueError('Frozen candidate code changed: ' + path)
    files = lock['model_files_sha256'][model]
    if not any(path.endswith('.safetensors') for path in files):
        raise ValueError('Model lock does not include weights')
    for path, sha in files.items():
        if digest(path) != sha:
            raise ValueError('Frozen model file changed: ' + path)
    return lock
