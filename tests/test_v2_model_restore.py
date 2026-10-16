import hashlib
import pytest
from scripts.prepare_v2_local_model import prepare


def record():
    return {'local_path': 'artifacts/v2/example', 'repo': 'example/model',
            'revision': 'frozen-revision', 'files': {
                'model.safetensors': {'bytes': 4, 'sha256': hashlib.sha256(b'good').hexdigest()}}}


def test_restore_uses_exact_revision_and_verifies_download(tmp_path):
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        kwargs['local_dir'].mkdir(parents=True)
        (kwargs['local_dir'] / kwargs['filename']).write_bytes(b'good')
    result = prepare(record(), tmp_path, download=download)
    assert calls[0]['revision'] == 'frozen-revision'
    assert result['downloaded_files'] == 1
    assert prepare(record(), tmp_path, verify_only=True)['downloaded_files'] == 0


def test_different_existing_weight_is_preserved_and_never_downloaded(tmp_path):
    target = tmp_path / 'artifacts/v2/example/model.safetensors'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'keep')
    with pytest.raises(ValueError, match='Existing model file differs'):
        prepare(record(), tmp_path, download=lambda **kw: pytest.fail('Must not download'))
    assert target.read_bytes() == b'keep'


def test_missing_weights_cannot_pass_verify_only(tmp_path):
    with pytest.raises(ValueError, match='Missing model files'):
        prepare(record(), tmp_path, verify_only=True)


def test_corrupt_download_is_not_reported_as_success(tmp_path):
    def download(**kwargs):
        kwargs['local_dir'].mkdir(parents=True)
        (kwargs['local_dir'] / kwargs['filename']).write_bytes(b'bad!')
    with pytest.raises(ValueError, match='frozen hash check'):
        prepare(record(), tmp_path, download=download)
