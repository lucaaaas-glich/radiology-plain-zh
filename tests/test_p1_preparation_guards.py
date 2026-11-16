"""Regression tests for staging boundaries; all downloads use in-memory fakes."""
import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts import prepare_p1_base as base
from scripts.prepare_p1_datasets import new_output_path, verify_quality_inputs


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('Tests must not access the network')
    monkeypatch.setattr(base.urllib.request, 'urlopen', refuse)


def spec_for(contents):
    return {'size': len(contents), 'lfs_sha256': hashlib.sha256(contents).hexdigest(),
            'git_blob_sha1': None}


def test_screen_binding_includes_target_and_full_risk_records(tmp_path):
    train = tmp_path / 'train.zh.jsonl'
    risks = tmp_path / 'per_sample.jsonl'
    summary = tmp_path / 'quality_summary.json'
    original = {'report': 'unchanged report', 'reference_zh': 'original answer', 'teacher_accepted': True}
    train.write_text(json.dumps(original))
    risks.write_text(json.dumps({'id': 'same-id', 'flags': []}))
    lock = {'input': {'sha256': hashlib.sha256(train.read_bytes()).hexdigest()},
            'local_artifacts': {'per_sample.jsonl': {'sha256': hashlib.sha256(risks.read_bytes()).hexdigest()}}}
    summary.write_text(json.dumps(lock))
    verify_quality_inputs(summary, train, risks)
    train.write_text(json.dumps(dict(original, reference_zh='modified answer')))
    with pytest.raises(ValueError, match='Training input changed'):
        verify_quality_inputs(summary, train, risks)
    train.write_text(json.dumps(original))
    risks.write_text(json.dumps({'id': 'same-id', 'flags': ['different flag']}))
    with pytest.raises(ValueError, match='Per-sample risks changed'):
        verify_quality_inputs(summary, train, risks)


def test_data_and_summary_paths_reject_history_and_existing_temporary_files(tmp_path):
    root = tmp_path.resolve()
    for historical in ('results/pilot_v1/new.json', 'results/v2/new.json', 'results/v3/new.json'):
        with pytest.raises(ValueError, match='permitted new P0/P1'):
            new_output_path(Path(historical), root, ('results/p0', 'results/p1'), check_tmp=True)
    with pytest.raises(ValueError, match='permitted new P0/P1'):
        new_output_path(root / 'data/p0', root, ('data/p0',))
    target = root / 'results/p0/run/new.json'
    target.parent.mkdir(parents=True)
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text('previous result must survive')
    with pytest.raises(FileExistsError, match='temporary'):
        new_output_path(target, root, ('results/p0', 'results/p1'), check_tmp=True)
    assert temporary.read_text() == 'previous result must survive'


def test_output_symlinks_are_rejected_even_inside_the_allowed_bucket(tmp_path):
    root = tmp_path.resolve()
    real = root / 'results/p0/real'
    real.mkdir(parents=True)
    (root / 'results/p0/alias').symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks'):
        new_output_path(root / 'results/p0/alias/new.json', root, ('results/p0',), check_tmp=True)
    (root / 'results/p0/dangling.json').symlink_to(real / 'missing.json')
    with pytest.raises(ValueError, match='symlinks'):
        new_output_path(root / 'results/p0/dangling.json', root, ('results/p0',), check_tmp=True)


def test_asset_root_itself_and_history_symlink_are_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='direct child'):
        base.asset_directory(Path('artifacts/p1'))
    history = tmp_path / 'artifacts/pilot_v1'
    history.mkdir(parents=True)
    (tmp_path / 'artifacts/p1').symlink_to(history, target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks'):
        base.asset_directory(Path('artifacts/p1/base-model'))
    assert list(history.iterdir()) == []


def test_asset_existing_provenance_or_partial_symlink_are_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / 'artifacts/p1/base-model'
    out.mkdir(parents=True)
    provenance = out / 'p1_model_provenance.json'
    provenance.write_text('{}')
    with pytest.raises(FileExistsError, match='prepared model'):
        base.asset_directory(out)
    provenance.unlink()
    (out / 'model.safetensors.download-partial').symlink_to(tmp_path / 'historical.bin')
    with pytest.raises(ValueError, match='symlinks'):
        base.asset_directory(out)


def test_completed_valid_partial_is_used_without_network_or_rehash(tmp_path, monkeypatch):
    contents = b'tiny test weights'
    destination = tmp_path / 'model.safetensors'
    partial = tmp_path / 'model.safetensors.download-partial'
    partial.write_bytes(contents)
    calls = []
    checksum = base.checksum

    def count_checksum(path, spec):
        calls.append(path)
        return checksum(path, spec)

    monkeypatch.setattr(base, 'checksum', count_checksum)
    assert base.download_file(destination, spec_for(contents)) == hashlib.sha256(contents).hexdigest()
    assert destination.read_bytes() == contents and not partial.exists()
    assert calls == [partial]


def test_promoting_partial_never_replaces_existing_file(tmp_path):
    contents = b'new contents'
    destination = tmp_path / 'model.safetensors'
    partial = tmp_path / 'model.safetensors.download-partial'
    destination.write_bytes(b'old historical contents')
    partial.write_bytes(contents)
    with pytest.raises(FileExistsError):
        base.install_verified_partial(partial, destination, spec_for(contents))
    assert destination.read_bytes() == b'old historical contents'
    assert partial.read_bytes() == contents


class FakeResponse(io.BytesIO):
    def __init__(self, content, status=200, headers=None):
        super().__init__(content)
        self.status = status
        self.headers = headers or {}


def test_partial_download_resumes_with_verified_range(tmp_path, monkeypatch):
    contents = b'original pinned model contents'
    destination = tmp_path / 'model.safetensors'
    partial = tmp_path / 'model.safetensors.download-partial'
    partial.write_bytes(contents[:8])
    requests = []

    def response(request, timeout):
        requests.append(request)
        return FakeResponse(contents[8:], 206, {'Content-Range': f'bytes 8-{len(contents)-1}/{len(contents)}'})

    monkeypatch.setattr(base.urllib.request, 'urlopen', response)
    base.download_file(destination, spec_for(contents))
    assert len(requests) == 1 and requests[0].headers['Range'] == 'bytes=8-'
    assert destination.read_bytes() == contents


def test_zero_length_interrupted_partial_can_resume(tmp_path, monkeypatch):
    contents = b'original pinned bytes'
    destination = tmp_path / 'config.json'
    (tmp_path / 'config.json.download-partial').touch()
    monkeypatch.setattr(base.urllib.request, 'urlopen', lambda *args, **kwargs: FakeResponse(contents))
    base.download_file(destination, spec_for(contents))
    assert destination.read_bytes() == contents


def test_ignored_range_or_incorrect_complete_partial_preserve_input(tmp_path, monkeypatch):
    contents = b'original pinned bytes'
    destination = tmp_path / 'model.safetensors'
    partial = tmp_path / 'model.safetensors.download-partial'
    partial.write_bytes(contents[:4])
    monkeypatch.setattr(base.urllib.request, 'urlopen', lambda *args, **kwargs: FakeResponse(contents))
    with pytest.raises(RuntimeError, match='Pinned asset download failed'):
        base.download_file(destination, spec_for(contents))
    assert partial.read_bytes() == contents[:4] and not destination.exists()
    partial.write_bytes(b'x' * len(contents))
    with pytest.raises(ValueError, match='LFS hash mismatch'):
        base.download_file(destination, spec_for(contents))
    assert partial.read_bytes() == b'x' * len(contents)
