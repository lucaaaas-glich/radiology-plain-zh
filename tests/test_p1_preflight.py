"""P1's meaningful CPU contracts, without torch, PEFT, weights or network."""
import json
import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from radplain import p1_contract, train_p1
from radplain.p1_contract import (
    MODEL_ID, MODEL_REVISION, PROVENANCE_NAME, collate_lists, encode_example,
    prepare_data, safe_output_path, sha256_file, validate_config, verify_original_snapshot,
)


class ToyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=False):
        assert return_dict is False
        ids = []
        for message in messages:
            ids += list(f"<{message['role']}>{message['content']}".encode()) + [2, 10]
        if add_generation_prompt:
            ids += list(b"<assistant>")
        return ids

    def decode(self, tokens, skip_special_tokens=False):
        return bytes(tokens).decode()


@pytest.fixture
def config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/p1_pilot.json").read_text())


@pytest.fixture(autouse=True)
def isolate_upstream_test_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(p1_contract, "UPSTREAM_LOCK_PATH", tmp_path / "upstream-test.lock.json")


def write_data(folder, train=96, validation=32, extras=None):
    folder.mkdir(parents=True)
    for split, count in (("train", train), ("validation", validation)):
        rows = [{"id": f"{split}-{index}", "report": f"{split} report {index}.",
                 "reference_zh": "没有气胸。", "teacher_accepted": True} for index in range(count)]
        rows.extend((extras or {}).get(split, []))
        (folder / f"{split}.zh.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return folder


def write_lock(folder, extra_files=None):
    folder.mkdir(parents=True, exist_ok=True)
    files = {"config.json": json.dumps({"model_type": "qwen2"}),
             "tokenizer.json": "{}", "tokenizer_config.json": "{}"}
    files.update(extra_files or {})
    for name, contents in files.items():
        (folder / name).write_text(contents, encoding="utf-8")
    upstream = {"model": MODEL_ID, "revision": MODEL_REVISION, "kind": "original_base", "files": {}}
    for name in files:
        contents = (folder / name).read_bytes()
        upstream["files"][name] = {
            "size": len(contents),
            "git_blob_sha1": hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\0" + contents).hexdigest(),
            "lfs_sha256": sha256_file(folder / name) if name.endswith(".safetensors") else None,
        }
    p1_contract.UPSTREAM_LOCK_PATH.write_text(json.dumps(upstream), encoding="utf-8")
    lock = {"model": MODEL_ID, "revision": MODEL_REVISION, "kind": "original_base",
            "files_sha256": {name: sha256_file(folder / name) for name in files},
            "upstream_lock_sha256": sha256_file(p1_contract.UPSTREAM_LOCK_PATH)}
    (folder / PROVENANCE_NAME).write_text(json.dumps(lock), encoding="utf-8")
    return folder


def test_modules_import_without_loading_torch_or_peft():
    code = """
import sys
import importlib.abc
class NoTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'peft', 'transformers'}:
            raise AssertionError('Training or tokenizer library imported eagerly: ' + fullname)
sys.meta_path.insert(0, NoTrainingImports())
import radplain.p1_contract
import radplain.train_p1
assert 'torch' not in sys.modules
assert 'peft' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_answer_and_eos_only_not_prompt_or_template_whitespace():
    row = {"report": "No pneumothorax.", "reference_zh": "没有气胸。"}
    encoded = encode_example(ToyTokenizer(), row, 10000)
    expected = list(row["reference_zh"].encode()) + [ToyTokenizer.eos_token_id]
    assert [value for value in encoded["labels"] if value != -100] == expected
    assert encoded["input_ids"][-1] == ToyTokenizer.eos_token_id
    assert encoded["template_trailing_tokens_ignored"] == 1
    assert all(value == -100 for value in encoded["labels"][:encoded["prompt_tokens"]])


def test_long_examples_are_rejected_without_truncation():
    assert encode_example(ToyTokenizer(), {"report": "report", "reference_zh": "answer"}, 3) is None


def test_prefix_mismatch_and_unexpected_post_eos_content_are_rejected():
    class BrokenPrefix(ToyTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=False):
            values = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return [99] + values if add_generation_prompt else values

    class BrokenSuffix(ToyTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=False):
            values = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return values if add_generation_prompt else values + [65]

    row = {"report": "report", "reference_zh": "answer"}
    with pytest.raises(ValueError, match="prefix mismatch"):
        encode_example(BrokenPrefix(), row, 10000)
    with pytest.raises(ValueError, match="after answer EOS"):
        encode_example(BrokenSuffix(), row, 10000)


def test_empty_answer_and_multiple_eos_are_rejected():
    with pytest.raises(ValueError, match="nonempty reference"):
        encode_example(ToyTokenizer(), {"report": "report", "reference_zh": " "}, 10000)
    with pytest.raises(ValueError, match="exactly one EOS"):
        encode_example(ToyTokenizer(), {"report": "report", "reference_zh": "answer\x02"}, 10000)


def test_template_dict_is_not_silently_converted_to_keys():
    class DictTokenizer(ToyTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [1, 2, 3]}

    with pytest.raises(ValueError, match="flat list of integer token IDs"):
        encode_example(DictTokenizer(), {"report": "report", "reference_zh": "answer"}, 10000)


def test_padding_is_masked_in_attention_and_loss():
    result = collate_lists([{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
                            {"input_ids": [1, 4], "labels": [-100, 4]}], 0)
    assert result["input_ids"][1] == [1, 4, 0, 0, 0, 0, 0, 0]
    assert result["attention_mask"][1] == [1, 1, 0, 0, 0, 0, 0, 0]
    assert result["labels"][1] == [-100, 4, -100, -100, -100, -100, -100, -100]


def test_pilot_real_encoding_plan_and_no_test_file_access(tmp_path, config):
    data = write_data(tmp_path / "data")
    (data / "test.zh.jsonl").write_text("THIS MUST NOT BE READ")
    encoded, report = prepare_data(ToyTokenizer(), config, data, "pilot")
    assert len(encoded["train"]) == 96 and len(encoded["validation"]) == 32
    assert report["plan"]["optimizer_steps"] == 6
    assert report["plan"]["effective_batch_size"] == 16
    assert report["splits"]["train"]["supervised_answer_eos_tokens"] == 96 * (len("没有气胸。".encode()) + 1)
    assert not report["final_test_read"] and not report["gpu_training_verified"]
    assert report["purpose"] == "engineering_pilot_not_formal_effectiveness"


def test_pilot_selection_is_deterministic_and_exclusions_explicit(tmp_path, config):
    extras = {"train": [
        {"id": "rejected", "teacher_accepted": False},
        {"id": "long", "report": "long" * 5000, "reference_zh": "answer", "teacher_accepted": True},
    ]}
    data = write_data(tmp_path / "data", train=128, extras=extras)
    first = prepare_data(ToyTokenizer(), config, data, "pilot")[1]
    second = prepare_data(ToyTokenizer(), config, data, "pilot")[1]
    assert first == second
    assert first["splits"]["train"]["pilot_not_selected_rows"] == 32
    assert [item["reason"] for item in first["splits"]["train"]["exclusions"]] == [
        "teacher_rejected", "too_long_no_truncation"]


def test_partial_accumulation_step_accounting(tmp_path, config):
    config = dict(config, pilot_train_size=65, expected_optimizer_steps=5)
    report = prepare_data(ToyTokenizer(), config, write_data(tmp_path / "data", train=65), "pilot")[1]
    assert report["plan"]["train_microbatches_per_epoch"] == 17
    assert report["plan"]["last_update_microbatches"] == 1
    assert report["plan"]["last_update_examples"] == 1


def test_formal_sft_cannot_use_pilot_size(tmp_path, config):
    with pytest.raises(ValueError, match="need 512 for sft"):
        prepare_data(ToyTokenizer(), config, write_data(tmp_path / "data"), "sft")


def test_formal_sft_uses_all_512_and_dev_32(tmp_path, config):
    config = dict(config, expected_optimizer_steps=32)
    encoded, report = prepare_data(ToyTokenizer(), config, write_data(tmp_path / "data", train=512), "sft")
    assert len(encoded["train"]) == 512 and report["plan"]["optimizer_steps"] == 32


def test_dev_minimum_and_duplicate_report_protection(tmp_path, config):
    data = write_data(tmp_path / "small_dev", validation=31)
    with pytest.raises(ValueError, match="need 32"):
        prepare_data(ToyTokenizer(), config, data, "pilot")
    data = write_data(tmp_path / "overlap")
    rows = [json.loads(line) for line in (data / "validation.zh.jsonl").read_text().splitlines()]
    rows[0]["report"] = "train report 0."
    (data / "validation.zh.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="Duplicate accepted report"):
        prepare_data(ToyTokenizer(), config, data, "pilot")


@pytest.mark.parametrize("change", [
    {"gradient_accumulation_steps": 0}, {"expected_effective_batch_size": 12},
    {"pilot_train_size": 63}, {"pilot_train_size": 129}, {"learning_rate": float("nan")},
    {"model_revision": "main"},
])
def test_invalid_configuration_is_rejected(config, change):
    with pytest.raises(ValueError):
        validate_config(dict(config, **change), "pilot")


def test_expected_optimizer_steps_checked_against_data(tmp_path, config):
    with pytest.raises(ValueError, match="optimizer steps"):
        prepare_data(ToyTokenizer(), dict(config, expected_optimizer_steps=7), write_data(tmp_path / "data"), "pilot")


def test_snapshot_tokenizer_hash_and_revision_checks(tmp_path):
    snapshot = write_lock(tmp_path / "snapshot")
    assert verify_original_snapshot(snapshot)["weights_verified"] is False
    (snapshot / "tokenizer.json").write_text("modified")
    with pytest.raises(ValueError, match="modified locked snapshot"):
        verify_original_snapshot(snapshot)
    write_lock(snapshot)
    lock = json.loads((snapshot / PROVENANCE_NAME).read_text())
    lock["revision"] = "main"
    (snapshot / PROVENANCE_NAME).write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="original Qwen base revision"):
        verify_original_snapshot(snapshot)


def test_complete_weights_and_index_must_be_locked(tmp_path):
    snapshot = write_lock(tmp_path / "snapshot", {"model.safetensors": "fake test weight bytes"})
    assert verify_original_snapshot(snapshot, require_weights=True)["weights_verified"]
    (snapshot / "model-extra.safetensors").write_text("unlocked shard")
    with pytest.raises(ValueError, match="complete safetensors index"):
        verify_original_snapshot(snapshot, require_weights=True)


@pytest.mark.parametrize("filename,content", [("tokenizer.json", "[]"), ("model.safetensors", "changed")])
def test_rehashed_local_provenance_cannot_hide_upstream_tampering(tmp_path, filename, content):
    snapshot = write_lock(tmp_path / "snapshot", {"model.safetensors": "weights"})
    (snapshot / filename).write_text(content)
    local_lock = json.loads((snapshot / PROVENANCE_NAME).read_text())
    local_lock["files_sha256"][filename] = sha256_file(snapshot / filename)
    (snapshot / PROVENANCE_NAME).write_text(json.dumps(local_lock))
    with pytest.raises(ValueError, match="upstream Git/LFS"):
        verify_original_snapshot(snapshot, require_weights=filename.endswith(".safetensors"))


def test_upstream_lock_binding_and_cpu_does_not_read_weights(tmp_path, monkeypatch):
    snapshot = write_lock(tmp_path / "snapshot", {"model.safetensors": "weights"})
    original_open = Path.open

    def reject_weight_read(path, *args, **kwargs):
        if path.name.endswith(".safetensors"):
            raise AssertionError("CPU preflight must never read weights")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_weight_read)
    assert verify_original_snapshot(snapshot)["upstream_git_lfs_verified"]
    p1_contract.UPSTREAM_LOCK_PATH.write_text(p1_contract.UPSTREAM_LOCK_PATH.read_text() + "\n")
    with pytest.raises(ValueError, match="not bound"):
        verify_original_snapshot(snapshot)


def test_every_weight_shard_must_be_in_local_provenance(tmp_path):
    snapshot = write_lock(tmp_path / "sharded", {
        "model-00001-of-00002.safetensors": "a", "model-00002-of-00002.safetensors": "b",
        "model.safetensors.index.json": json.dumps({"weight_map": {
            "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}),
    })
    lock = json.loads((snapshot / PROVENANCE_NAME).read_text())
    del lock["files_sha256"]["model-00002-of-00002.safetensors"]
    (snapshot / PROVENANCE_NAME).write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="does not cover all required files"):
        verify_original_snapshot(snapshot, require_weights=True)


def test_merged_adapter_and_quantized_bases_rejected(tmp_path):
    with pytest.raises(ValueError, match="Merged or LoRA"):
        verify_original_snapshot(write_lock(tmp_path / "merged-model"))
    snapshot = write_lock(tmp_path / "snapshot")
    (snapshot / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="Adapter"):
        verify_original_snapshot(snapshot)
    snapshot = write_lock(tmp_path / "quantized", {"config.json": json.dumps(
        {"model_type": "qwen2", "quantization_config": {"load_in_4bit": True}})})
    with pytest.raises(ValueError, match="unquantized"):
        verify_original_snapshot(snapshot)


def test_output_protects_existing_history_wrong_bucket_and_nested_runs(tmp_path):
    fresh = tmp_path / "results/p1/fresh"
    assert safe_output_path(fresh, "preflight", project_root=tmp_path) == fresh
    for path in (tmp_path / "results/pilot_v1/new", tmp_path / "results/v3/new",
                 tmp_path / "artifacts/p1/new", tmp_path / "results/p1/old/nested"):
        with pytest.raises(ValueError, match="new direct run directory"):
            safe_output_path(path, "preflight", project_root=tmp_path)
    fresh.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        safe_output_path(fresh, "preflight", project_root=tmp_path)


def test_output_symlink_cannot_redirect_to_history(tmp_path):
    history = tmp_path / "results/pilot_v1"
    history.mkdir(parents=True)
    (tmp_path / "results/p1").symlink_to(history, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        safe_output_path(tmp_path / "results/p1/new", "preflight", project_root=tmp_path)


def test_cpu_preflight_writes_new_report_without_model_loading(tmp_path, config, monkeypatch):
    monkeypatch.setattr(p1_contract, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(train_p1, "load_tokenizer", lambda path: ToyTokenizer())
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    data = write_data(tmp_path / "data")
    snapshot = write_lock(tmp_path / "tokenizer-only")
    output = tmp_path / "results/p1/test"
    report = train_p1.preflight(config_file, data, snapshot, output, "pilot")
    assert json.loads((output / "preflight.json").read_text()) == report
    assert not report["tokenizer_provenance"]["weights_verified"]
    assert not list(snapshot.glob("*.safetensors"))
    with pytest.raises(FileExistsError):
        train_p1.preflight(config_file, data, snapshot, output, "pilot")


def test_train_without_cuda_fails_before_creating_outputs(tmp_path, config, monkeypatch):
    monkeypatch.setattr(p1_contract, "PROJECT_ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    output = tmp_path / "artifacts/p1/test"
    with pytest.raises(RuntimeError, match="scheduled CUDA allocation"):
        train_p1.train(config_file, "unused-data", "unused-tokenizer", "unused-model", output, "pilot")
    assert not output.exists()


def test_cli_requires_explicit_mode_and_training_model():
    with pytest.raises(SystemExit):
        train_p1.main(["preflight", "--data", "data", "--tokenizer", "tokenizer", "--out", "out"])
    with pytest.raises(SystemExit):
        train_p1.main(["train", "--mode", "pilot", "--data", "data", "--tokenizer", "tokenizer", "--out", "out"])
