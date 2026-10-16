import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from scripts.v2_local_preflight import validate


def write(path, value):
    Path(path).write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def git(*args):
        return subprocess.check_output(['git', *args], stderr=subprocess.DEVNULL).decode().strip()
    git('init')
    git('config', 'user.name', 'Test')
    git('config', 'user.email', 'test@example.invalid')
    Path('weights.safetensors').write_bytes(b'fake weights: no model is loaded')
    Path('source.py').write_text('frozen = True\n')
    Path('input.jsonl').write_text('{}\n')
    lock = {'arms': {'a': {'model': 'model', 'mode': 'direct', 'max_tokens': 64}},
            'code_sha256': {'source.py': sha('source.py')},
            'holdout_sha256': {'synthetic': sha('input.jsonl')},
            'model_files_sha256': {'model': {'weights.safetensors': sha('weights.safetensors')}}}
    write('candidate.json', lock)
    git('add', 'source.py', 'candidate.json')
    git('commit', '-m', 'freeze candidate')
    barrier = {'state': 'released', 'released_at_utc': 'test',
               'local_candidate_lock': 'candidate.json',
               'local_candidate_lock_sha256': sha('candidate.json'),
               'local_candidate_commit': git('rev-parse', 'HEAD')}
    def save_barrier(value):
        write('barrier.json', value)
        git('add', 'barrier.json')
        git('commit', '-m', 'record barrier')
    save_barrier(barrier)
    def check():
        return validate('candidate.json', 'a', 'model', 'direct', 64,
                        'input.jsonl', 'holdout', 'barrier.json')
    return check, barrier, save_barrier


def test_preflight_accepts_only_bound_committed_release(frozen):
    check, _, _ = frozen
    assert check()['arms']['a']['model'] == 'model'


def test_preflight_rejects_held_barrier_before_model_loading(frozen):
    check, barrier, save = frozen
    barrier['state'] = 'held'
    save(barrier)
    with pytest.raises(ValueError, match='still held'):
        check()


def test_preflight_rejects_rebound_lock(frozen):
    check, barrier, save = frozen
    barrier['local_candidate_lock_sha256'] = 'wrong'
    save(barrier)
    with pytest.raises(ValueError, match='not bound'):
        check()


def test_preflight_rejects_changed_weights(frozen):
    check, _, _ = frozen
    Path('weights.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='model file changed'):
        check()


def test_preflight_rejects_uncommitted_inference_code(frozen):
    check, _, _ = frozen
    Path('source.py').write_text('frozen = False\n')
    with pytest.raises(subprocess.CalledProcessError):
        check()
