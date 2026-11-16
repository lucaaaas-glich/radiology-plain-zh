"""Source locks, arm identity, private journals and interruption recovery on CPU."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from radplain import evaluate_p1 as evaluation


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def freeze_sources(tmp_path, rows=None):
    rows = rows if rows is not None else [{"id": str(index), "report": f"source report {index}"} for index in range(3)]
    sources, lock_path = tmp_path / "sources.jsonl", tmp_path / "sources.lock.json"
    sources.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    locked = {"schema": "p1-source-lock-v1", "purpose": evaluation.PURPOSE, "decode": evaluation.DECODE,
              "source_file_sha256": evaluation.sha256_file(sources),
              "rows": [{"id": row["id"], "source_sha256": hashlib.sha256(row["report"].encode()).hexdigest()} for row in rows],
              "training_config_sha256": "config-hash", "training_inputs_sha256": {"S0": "S0-train", "S1": "S1-train"}}
    write_json(lock_path, locked)
    return sources, lock_path, locked


def terminal(row, status="completed"):
    return dict(row, status=status, prediction="中文生成" if status == "completed" else "", error_code=None if status == "completed" else "test_failure",
                token_ids=[10, 2] if status == "completed" else [],
                generation={"input_tokens": 10, "output_tokens": 2 if status == "completed" else 0,
                            "eos_completed": status == "completed", "elapsed_seconds": 0.25})


@pytest.fixture
def sources(tmp_path):
    return freeze_sources(tmp_path)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "PROJECT_ROOT", tmp_path)
    provenance = {"verified_files_sha256": {"config.json": "config", "model.safetensors": "original-model"}}
    monkeypatch.setattr(evaluation, "verify_original_snapshot", lambda *args, **kwargs: provenance)
    cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                           max_memory_allocated=lambda: 123, max_memory_reserved=lambda: 456)
    monkeypatch.setattr(evaluation, "require_cuda", lambda: SimpleNamespace(cuda=cuda))
    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1, 2, 3])
    monkeypatch.setattr(evaluation, "load_runtime", lambda *args: (object(), tokenizer, object()))
    calls = []

    def generate(*args):
        calls.append(1)
        row = terminal({})
        return row

    monkeypatch.setattr(evaluation, "generate_one", generate)
    return provenance, calls


def test_module_import_does_not_load_model_libraries():
    code = """
import sys, importlib.abc
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'peft', 'safetensors'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, NoModels())
import radplain.evaluate_p1
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_frozen_source_only_identity_and_file_hash(sources):
    source, lock_path, lock = sources
    rows, _, hashes = evaluation.load_sources(source, lock_path)
    assert len(rows) == 3 and hashes["sources_sha256"] == lock["source_file_sha256"]
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen hash"):
        evaluation.load_sources(source, lock_path)


@pytest.mark.parametrize("rows", [
    [{"id": "a", "report": "one"}, {"id": "a", "report": "two"}],
    [{"id": "a", "report": "duplicate"}, {"id": "b", "report": " DUPLICATE "}],
    [{"id": "a", "report": "one", "reference_zh": "must not be read as a target"}],
    [{"id": "a", "report": "one", "source_sha256": "changed"}],
])
def test_duplicate_or_target_contaminated_sources_are_rejected(tmp_path, rows):
    source, lock_path, _ = freeze_sources(tmp_path, rows)
    with pytest.raises(ValueError):
        evaluation.load_sources(source, lock_path)


def test_lock_order_and_decode_are_fixed(sources):
    source, lock_path, lock = sources
    modified = copy.deepcopy(lock)
    modified["rows"].reverse()
    write_json(lock_path, modified)
    with pytest.raises(ValueError, match="identities/order"):
        evaluation.load_sources(source, lock_path)
    modified = copy.deepcopy(lock)
    modified["decode"] = dict(evaluation.DECODE, max_new_tokens=256)
    write_json(lock_path, modified)
    with pytest.raises(ValueError, match="input/decode lock"):
        evaluation.load_sources(source, lock_path)


def test_base_arm_cannot_accept_an_adapter_or_epoch():
    for adapter, epoch in (("old-adapter", None), (None, 1), ("old-adapter", 2)):
        with pytest.raises(ValueError, match="must not use"):
            evaluation.adapter_identity("base", adapter, epoch, {}, {})


@pytest.fixture
def saved_sft(tmp_path, monkeypatch):
    cfg = {"student_model": evaluation.MODEL_ID, "model_revision": evaluation.MODEL_REVISION, "seed": 7,
           "max_sequence_length": 1152, "epochs": 2, "train_batch_size": 4, "gradient_accumulation_steps": 4,
           "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.05, "learning_rate": 0.0001}
    history = [{"epoch": 1, "validation_answer_token_loss": 1.2}, {"epoch": 2, "validation_answer_token_loss": 1.1}]
    provenance = {"model": evaluation.MODEL_ID, "revision": evaluation.MODEL_REVISION,
                  "verified_files_sha256": {"model.safetensors": "original-model"}}
    preflight = {"mode": "sft", "config": cfg, "config_source_sha256": "config-hash",
                 "splits": {"train": {"used_rows": 1024, "source_sha256": "S0-train"}, "validation": {"used_rows": 40}},
                 "plan": {"optimizer_steps": 128}}
    training = {"mode": "sft", "training_completed": True, "final_test_read": False, "config": cfg,
                "history": history, "optimizer_steps": 128, "nonzero_gradient_steps": 128}
    run = tmp_path / "sft"
    for name, record in {"config": cfg, "training": training, "preflight": preflight,
                         "history": history, "model_provenance": provenance}.items():
        write_json(run / f"{name}.json", record)
    for epoch in (1, 2):
        path = run / f"epoch-{epoch}"
        write_json(path / "epoch_metrics.json", history[epoch - 1])
        write_json(path / "adapter_config.json", {"peft_type": "LORA", "task_type": "CAUSAL_LM", "bias": "none",
                   "r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]})
    monkeypatch.setattr(evaluation, "inspect_adapter_weights", lambda path: {"sha256": "adapter-weights"})
    lock = {"training_config_sha256": "config-hash", "training_inputs_sha256": {"S0": "S0-train", "S1": "S1-train"}}
    return run, lock, provenance


def test_each_saved_sft_epoch_and_arm_data_are_bound(saved_sft):
    run, lock, provenance = saved_sft
    for epoch in (1, 2):
        assert evaluation.adapter_identity("S0", run / f"epoch-{epoch}", epoch, lock, provenance)["training_data_sha256"] == "S0-train"
    with pytest.raises(ValueError, match="arm-specific data"):
        evaluation.adapter_identity("S1", run / "epoch-1", 1, lock, provenance)
    with pytest.raises(ValueError, match="epoch-N"):
        evaluation.adapter_identity("S0", run / "epoch-1", 2, lock, provenance)
    lock["training_config_sha256"] = "different-config"
    with pytest.raises(ValueError, match="training config"):
        evaluation.adapter_identity("S0", run / "epoch-1", 1, lock, provenance)


def test_incomplete_sft_or_mismatched_base_is_rejected(saved_sft):
    run, lock, provenance = saved_sft
    changed = dict(provenance, verified_files_sha256={"model.safetensors": "different"})
    with pytest.raises(ValueError, match="exact locked original base"):
        evaluation.adapter_identity("S0", run / "epoch-1", 1, lock, changed)
    (run / "failure.json").write_text("{}")
    with pytest.raises(ValueError, match="completed"):
        evaluation.adapter_identity("S0", run / "epoch-1", 1, lock, provenance)


def test_journal_recovery_preserves_success_and_failure_and_only_drops_partial_tail(tmp_path, sources):
    rows = evaluation.load_sources(sources[0], sources[1])[0]
    path = tmp_path / "predictions.jsonl"
    path.touch()
    records = []
    evaluation.append_result(path, terminal(rows[0]), records, "identity")
    evaluation.append_result(path, terminal(rows[1], "failed"), records, "identity")
    prefix = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'{"id":"2","half_written":')
    recovered = evaluation.read_journal(path, rows, "identity", recover=True)
    assert path.read_bytes() == prefix and recovered == records
    assert [row["id"] for row in evaluation.pending_rows(rows, recovered)] == ["2"]
    backups = list(tmp_path.glob("recovery-tail-*.bin"))
    assert len(backups) == 1 and backups[0].read_bytes() == b'{"id":"2","half_written":'
    with pytest.raises(ValueError, match="already terminal"):
        evaluation.append_result(path, terminal(rows[1]), recovered, "identity")


def test_complete_missing_newline_is_preserved_without_rerunning(tmp_path, sources):
    rows = evaluation.load_sources(sources[0], sources[1])[0]
    path = tmp_path / "predictions.jsonl"
    path.touch()
    records = []
    evaluation.append_result(path, terminal(rows[0]), records, "identity")
    path.write_bytes(path.read_bytes()[:-1])
    recovered = evaluation.read_journal(path, rows, "identity", recover=True)
    assert len(recovered) == 1 and path.read_bytes().endswith(b"\n")


def test_complete_corrupt_or_duplicate_journal_is_never_repaired(tmp_path, sources):
    rows = evaluation.load_sources(sources[0], sources[1])[0]
    path = tmp_path / "predictions.jsonl"
    path.touch()
    records = []
    evaluation.append_result(path, terminal(rows[0]), records, "identity")
    corrupt = path.read_bytes().replace("中文生成".encode(), "修改结果".encode())
    path.write_bytes(corrupt)
    with pytest.raises(ValueError, match="hash chain"):
        evaluation.read_journal(path, rows, "identity", recover=True)
    assert path.read_bytes() == corrupt


def test_evaluation_resume_never_repeats_terminal_failure(tmp_path, sources, runtime, monkeypatch):
    source, lock_path, _ = sources
    calls = []

    def interrupted(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("private source detail must stay out of public output")
        return terminal({})

    monkeypatch.setattr(evaluation, "generate_one", interrupted)
    args = ("base", source, lock_path, "base-model", "data/p1/run", "results/p1/run")
    first = evaluation.evaluate(*args)
    assert first["counts"] == {"requested": 3, "terminal": 2, "completed": 1, "failed": 1, "pending": 1, "eos_completed": 1}
    original = (tmp_path / "data/p1/run/predictions.jsonl").read_bytes()
    monkeypatch.setattr(evaluation, "generate_one", lambda *args: terminal({}))
    second = evaluation.evaluate(*args, resume=True)
    assert second["status"] == "completed_with_failures" and second["counts"]["failed"] == 1
    assert (tmp_path / "data/p1/run/predictions.jsonl").read_bytes().startswith(original)
    assert len(list((tmp_path / "results/p1/run").glob("manifest-*.json"))) == 2
    text = "".join(path.read_text() for path in (tmp_path / "results/p1/run").glob("*.json"))
    assert "private source detail" not in text and "source report" not in text and "中文生成" not in text
    monkeypatch.setattr(evaluation, "require_cuda", lambda: (_ for _ in ()).throw(AssertionError("All IDs are already terminal")))
    assert evaluation.evaluate(*args, resume=True)["counts"]["terminal"] == 3


def test_output_overwrite_or_resume_identity_change_is_rejected(tmp_path, sources, runtime):
    source, lock_path, lock = sources
    args = ("base", source, lock_path, "base-model", "data/p1/run", "results/p1/run")
    evaluation.evaluate(*args)
    with pytest.raises(FileExistsError):
        evaluation.evaluate(*args)
    lock["note"] = "Even metadata change must invalidate the frozen input lock identity"
    write_json(lock_path, lock)
    with pytest.raises(ValueError, match="identity has changed"):
        evaluation.evaluate(*args, resume=True)


def test_overlong_prompt_is_terminal_failure_without_truncation(tmp_path, sources, runtime, monkeypatch):
    monkeypatch.setattr(evaluation, "encode_prompt", lambda *args: list(range(4097)))
    monkeypatch.setattr(evaluation, "generate_one", lambda *args: (_ for _ in ()).throw(AssertionError("Overlength inputs must not generate")))
    summary = evaluation.evaluate("base", sources[0], sources[1], "base", "data/p1/long", "results/p1/long")
    assert summary["counts"]["failed"] == 3 and summary["tokens"]["input_tokens"] == 3 * 4097
    rows = [json.loads(line) for line in (tmp_path / "data/p1/long/predictions.jsonl").read_text().splitlines()]
    assert all(row["error_code"] == "input_overlength_no_truncation" for row in rows)


def test_append_failure_does_not_append_a_second_record_into_an_interrupted_write(tmp_path, sources, runtime, monkeypatch):
    monkeypatch.setattr(evaluation, "encode_prompt", lambda *args: list(range(4097)))
    attempts = []

    def partial_write(path, *args):
        attempts.append(1)
        with path.open("ab") as stream:
            stream.write(b'{"interrupted":')
        raise OSError("simulated disk interruption")

    monkeypatch.setattr(evaluation, "append_result", partial_write)
    summary = evaluation.evaluate("base", sources[0], sources[1], "base", "data/p1/disk", "results/p1/disk")
    assert len(attempts) == 1 and summary["counts"]["terminal"] == 0
    assert (tmp_path / "data/p1/disk/predictions.jsonl").read_bytes() == b'{"interrupted":'


def test_raw_paths_cannot_escape_to_history_or_public(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "PROJECT_ROOT", tmp_path)
    for path in ("results/p1/raw", "data/pilot_v1/new", "data/p1/old/nested"):
        with pytest.raises(ValueError):
            evaluation.output_directory(path, "data/p1", False)
    existing = tmp_path / "data/p1/real"
    existing.mkdir(parents=True)
    (tmp_path / "data/p1/alias").symlink_to(existing, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        evaluation.output_directory("data/p1/alias", "data/p1", True)


def test_concurrent_writer_is_rejected(tmp_path):
    with evaluation.writer_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already writing"):
            with evaluation.writer_lock(tmp_path):
                pass
