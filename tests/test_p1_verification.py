"""Saved-adapter/probe contracts, runnable without torch, GPU or model loading."""
import copy
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from radplain.common import fingerprint
from radplain import verify_p1


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def pilot(tmp_path):
    run, data = tmp_path / "run", tmp_path / "data"
    run.mkdir()
    data.mkdir()
    rows = [{"id": f"dev-{i}", "report": f"report {i}", "reference_zh": "原始目标。", "teacher_accepted": True}
            for i in range(12)]
    validation = data / "validation.zh.jsonl"
    validation.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (data / "test.zh.jsonl").write_text("MUST NEVER BE READ")
    (data / "train.zh.jsonl").write_text("NOT NEEDED BY RELOAD VERIFICATION")
    config = {"lora_rank": 2, "lora_alpha": 4, "max_sequence_length": 1152}
    history = [{"epoch": 1, "validation_answer_token_loss": 1.2}]
    preflight = {"mode": "pilot", "final_test_read": False, "config": config,
                 "config_source_sha256": "fixed-config-hash", "plan": {"optimizer_steps": 6},
                 "splits": {"train": {"source_sha256": "fixed-train-hash"}, "validation": {
                     "source_sha256": verify_p1.sha256_file(validation),
                     "selected_rows": [{"row": i + 1, "id": row["id"], "report_sha256": fingerprint(row["report"])}
                                       for i, row in enumerate(rows)]}}}
    records = {"config": config, "preflight": preflight, "history": history,
               "training": {"mode": "pilot", "training_completed": True, "final_test_read": False,
                            "optimizer_steps": 6, "config": config, "history": history},
               "selection": {"mode": "pilot", "final_test_read": False, "epoch": 1,
                             "adapter_path": "epoch-1", "validation_answer_token_loss": 1.2},
               "model_provenance": {"model": verify_p1.MODEL_ID, "revision": verify_p1.MODEL_REVISION}}
    for name, value in records.items():
        write_json(run / f"{name}.json", value)
    write_json(run / "epoch-1/adapter_config.json", {"peft_type": "LORA", "task_type": "CAUSAL_LM", "bias": "none",
               "r": 2, "lora_alpha": 4, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]})
    plan = {"purpose": "engineering_save_reload_and_generation_smoke_only",
            "source": "old_development_only_not_final_test", "quality_claim": False,
            "model_predictions_seen": False, "arms": list(verify_p1.ARMS),
            "decode": {"do_sample": False, "max_new_tokens": 256},
            "validation_file_sha256": verify_p1.sha256_file(validation),
            "training_config_sha256": "fixed-config-hash", "training_data_sha256": "fixed-train-hash",
            "reports": [{"id": row["id"], "source_sha256": hashlib.sha256(row["report"].encode()).hexdigest()}
                        for row in rows[:10]]}
    return run, data, rows, records, plan


def test_import_does_not_load_torch_peft_transformers_or_safetensors():
    code = """
import sys, importlib.abc
class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'peft', 'transformers', 'safetensors'}:
            raise AssertionError('Eager model-library import: ' + fullname)
sys.meta_path.insert(0, NoModelImports())
import radplain.verify_p1
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_completed_pilot_reads_only_the_bound_development_rows(pilot, monkeypatch):
    run, data, rows, records, plan = pilot
    original_open = Path.open

    def guard(path, *args, **kwargs):
        if path.name in ("train.zh.jsonl", "test.zh.jsonl"):
            raise AssertionError("Probe must not read training or final test files")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    adapter, probes, loaded, hashes = verify_p1.completed_pilot(run, data)
    verify_p1.validate_generation_plan(plan, probes, loaded)
    assert probes == rows[:10] and adapter == run / "epoch-1"
    assert "development_file" in hashes


@pytest.mark.parametrize("file,key,value", [
    ("training", "training_completed", False), ("training", "optimizer_steps", 5),
    ("training", "mode", "sft"), ("selection", "adapter_path", "../old-adapter"),
    ("selection", "validation_answer_token_loss", 0.8),
])
def test_incomplete_or_mismatched_pilot_is_rejected(pilot, file, key, value):
    run, data, _, records, _ = pilot
    records[file][key] = value
    write_json(run / f"{file}.json", records[file])
    with pytest.raises(ValueError):
        verify_p1.completed_pilot(run, data)


def test_development_content_change_and_symlink_adapter_are_rejected(pilot, tmp_path):
    run, data, rows, records, _ = pilot
    original = (data / "validation.zh.jsonl").read_bytes()
    (data / "validation.zh.jsonl").write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="Development file differs"):
        verify_p1.completed_pilot(run, data)
    (data / "validation.zh.jsonl").write_bytes(original)
    (run / "epoch-1").rename(run / "elsewhere")
    (run / "epoch-1").symlink_to(run / "elsewhere", target_is_directory=True)
    with pytest.raises(ValueError, match="inside its pilot run"):
        verify_p1.completed_pilot(run, data)


@pytest.mark.parametrize("change", ["report_order", "decode", "train_hash", "selected_rows"])
def test_frozen_plan_rejects_probe_or_decode_drift(pilot, change):
    _, _, rows, records, plan = pilot
    if change == "report_order":
        plan["reports"].reverse()
    elif change == "decode":
        plan["decode"]["max_new_tokens"] = 384
    elif change == "train_hash":
        plan["training_data_sha256"] = "different"
    else:
        records["preflight"]["splits"]["validation"]["selected_rows"][0]["row"] = 2
    with pytest.raises(ValueError):
        verify_p1.validate_generation_plan(plan, rows[:10], records)


def save_fake_safetensors(path, dtype="F32", special_bits=None, empty=False):
    code, width, _ = verify_p1.FLOAT_BITS[dtype]
    prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    contents = struct.pack("<" + code, 0 if special_bits is None else special_bits) * 4
    header = {} if empty else {f"{prefix}.lora_A.weight": {"dtype": dtype, "shape": [2, 2], "data_offsets": [0, width * 4]},
                              f"{prefix}.lora_B.weight": {"dtype": dtype, "shape": [2, 2], "data_offsets": [width * 4, width * 8]}}
    encoded = json.dumps(header).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + (b"" if empty else contents * 2))


@pytest.mark.parametrize("dtype", ["F16", "BF16", "F32", "F64"])
def test_saved_adapter_finite_check_supports_float_formats_without_torch(tmp_path, dtype):
    path = tmp_path / "adapter_model.safetensors"
    save_fake_safetensors(path, dtype)
    inspected = verify_p1.inspect_adapter_weights(path)
    assert len(inspected["shapes"]) == 2 and inspected["sha256"] == verify_p1.sha256_file(path)


@pytest.mark.parametrize("dtype,delta", [("F16", 0), ("BF16", 1), ("F32", 0), ("F64", 1)])
def test_saved_adapter_nan_and_infinity_are_rejected(tmp_path, dtype, delta):
    path = tmp_path / "adapter_model.safetensors"
    save_fake_safetensors(path, dtype, verify_p1.FLOAT_BITS[dtype][2] + delta)
    with pytest.raises(ValueError, match="non-finite"):
        verify_p1.inspect_adapter_weights(path)


def test_empty_and_truncated_saved_weights_are_rejected(tmp_path):
    path = tmp_path / "adapter_model.safetensors"
    save_fake_safetensors(path, empty=True)
    with pytest.raises(ValueError, match="no tensors"):
        verify_p1.inspect_adapter_weights(path)
    save_fake_safetensors(path)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte range"):
        verify_p1.inspect_adapter_weights(path)


def test_reload_state_requires_exact_keys_and_shapes():
    tensor = SimpleNamespace(shape=(2, 4))
    verify_p1.same_tensor_structure({"A": tensor}, {"A": tensor})
    for changed in ({}, {"B": tensor}, {"A": SimpleNamespace(shape=(4, 2))}):
        with pytest.raises(ValueError, match="keys or shapes"):
            verify_p1.same_tensor_structure({"A": tensor}, changed)


def test_eos_and_chinese_are_separate_and_public_counts_have_no_raw_text():
    tokenizer = SimpleNamespace(eos_token_id=2, decode=lambda values, **kwargs: "中文输出" if 10 in values else "English")
    eos = verify_p1.generation_result(tokenizer, [10, 2, 99])
    capped = verify_p1.generation_result(tokenizer, [10])
    english = verify_p1.generation_result(tokenizer, [11, 2])
    assert eos["eos_completed"] and eos["contains_han"] and eos["token_ids"] == [10, 2]
    assert not capped["eos_completed"] and capped["contains_han"]
    assert english["eos_completed"] and not english["contains_han"]
    summary = verify_p1.completion_summary([eos, capped, english])
    assert summary == {"requested": 10, "generated": 3, "eos_completed": 2, "nonempty": 3, "contains_han": 2}
    assert "中文" not in json.dumps(summary) and "token_ids" not in summary


def test_raw_outputs_cannot_enter_public_or_existing_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_p1, "PROJECT_ROOT", tmp_path)
    assert verify_p1.raw_output_path("data/p1/fresh") == tmp_path / "data/p1/fresh"
    with pytest.raises(ValueError, match="ignored data/p1"):
        verify_p1.raw_output_path("results/p1/raw")
    target = tmp_path / "data/p1/existing"
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        verify_p1.raw_output_path(target)
    (tmp_path / "data/p1/alias").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        verify_p1.raw_output_path(tmp_path / "data/p1/alias")
