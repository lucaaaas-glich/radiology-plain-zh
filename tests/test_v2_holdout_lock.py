import json
import pytest
from scripts.run_v2_remote_holdout import digest,verify_model_manifest


def model_fixture(root):
    (root/'model-00001.safetensors').write_bytes(b'test bytes, not a model')
    (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'layer':'model-00001.safetensors'}}))
    hashes={p.name:digest(p) for p in root.iterdir()}
    return {'model_repo':'test/repo'}, {'model_files_sha256':{'test/repo':hashes}}, {'model_files_sha256':dict(hashes)}


def test_request_cannot_replace_frozen_model_manifest(tmp_path):
    arm,lock,request=model_fixture(tmp_path)
    request['model_files_sha256'].pop('model-00001.safetensors')
    with pytest.raises(ValueError,match='differs'):
        verify_model_manifest(request,lock,arm,tmp_path)


def test_index_cannot_reference_unhashed_weights(tmp_path):
    arm,lock,request=model_fixture(tmp_path)
    lock['model_files_sha256']['test/repo'].pop('model-00001.safetensors')
    request['model_files_sha256'].pop('model-00001.safetensors')
    with pytest.raises(ValueError,match='shard'):
        verify_model_manifest(request,lock,arm,tmp_path)


def test_changed_weights_rejected_and_intact_weights_accepted(tmp_path):
    arm,lock,request=model_fixture(tmp_path)
    verify_model_manifest(request,lock,arm,tmp_path)
    (tmp_path/'model-00001.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):
        verify_model_manifest(request,lock,arm,tmp_path)
