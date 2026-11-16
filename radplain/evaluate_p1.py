"""Resumable, locked P1 source-only generation; no automatic quality scoring.

Each source gets one durable terminal result, including failures. Explicit
--resume processes only IDs without a terminal record. An interrupted trailing
write is preserved before recovery; completed and failed rows are never rerun.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import traceback

from .common import fingerprint
from .p1_contract import MODEL_ID, MODEL_REVISION, PROJECT_ROOT, sha256_file, validate_config, verify_original_snapshot
from .prompts import PROMPT_VERSION, messages
from .train_p1 import load_tokenizer, require_cuda, write_new_json
from .verify_p1 import digest_json, generation_result, inspect_adapter_weights, same_tensor_structure

PURPOSE = "controlled_source_consistency_comparison_AI_review_not_clinical"
DECODE = {"do_sample": False, "max_new_tokens": 512, "max_input_tokens": 4096}


def strict_json(value):
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = item
        return result
    return json.loads(value, object_pairs_hook=unique)


def load_sources(source_path, lock_path):
    source_path, lock_path = Path(source_path), Path(lock_path)
    source_bytes = source_path.read_bytes()
    lock_bytes = lock_path.read_bytes()
    lock = strict_json(lock_bytes)
    if (lock.get("schema") != "p1-source-lock-v1" or lock.get("purpose") != PURPOSE
            or lock.get("decode") != DECODE):
        raise ValueError("Expected the fixed P1 source-only input/decode lock")
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != lock.get("source_file_sha256"):
        raise ValueError("Source file differs from its frozen hash")
    rows, ids, reports = [], set(), set()
    for line in source_bytes.splitlines():
        if not line.strip():
            continue
        row = strict_json(line)
        if not isinstance(row, dict) or set(row) - {"id", "report", "source_sha256"}:
            raise ValueError("Sources may contain only id/report/source_sha256, never reference targets")
        if not isinstance(row.get("id"), str) or not row["id"].strip() or row["id"] in ids:
            raise ValueError("Source IDs must be nonempty unique strings")
        if not isinstance(row.get("report"), str) or not row["report"].strip():
            raise ValueError("Source reports must be nonempty strings")
        source_sha = hashlib.sha256(row["report"].encode()).hexdigest()
        if "source_sha256" in row and row["source_sha256"] != source_sha:
            raise ValueError("Source row hash differs from the exact report bytes")
        normalized = fingerprint(row["report"])
        if normalized in reports:
            raise ValueError("Duplicate source reports are forbidden even under different IDs")
        ids.add(row["id"])
        reports.add(normalized)
        rows.append({"id": row["id"], "report": row["report"], "source_sha256": source_sha})
    expected = [{"id": row["id"], "source_sha256": row["source_sha256"]} for row in rows]
    if not rows or expected != lock.get("rows"):
        raise ValueError("Source identities/order differ from the frozen input lock")
    return rows, lock, {"sources_sha256": source_hash, "input_lock_sha256": hashlib.sha256(lock_bytes).hexdigest()}


def adapter_identity(arm, adapter, epoch, input_lock, base_provenance):
    if arm not in ("base", "S0", "S1"):
        raise ValueError("Expected base, S0 or S1")
    if arm == "base":
        if adapter is not None or epoch is not None:
            raise ValueError("The original base arm must not use an adapter or epoch")
        return None
    if adapter is None or epoch not in (1, 2):
        raise ValueError("S0/S1 require an explicit saved adapter and epoch 1 or 2")
    adapter = Path(adapter)
    if adapter.is_symlink() or adapter.name != f"epoch-{epoch}":
        raise ValueError("Adapter must be its original epoch-N directory, without a symlink")
    adapter = adapter.resolve(strict=True)
    run = adapter.parent
    records = {name: strict_json((run / f"{name}.json").read_bytes())
               for name in ("config", "training", "preflight", "history", "model_provenance")}
    cfg, training, preflight = (records[key] for key in ("config", "training", "preflight"))
    validate_config(cfg, "sft")
    if (training.get("mode") != "sft" or preflight.get("mode") != "sft"
            or training.get("training_completed") is not True or (run / "failure.json").exists()
            or training.get("final_test_read") is not False):
        raise ValueError("Adapter must come from a completed, development-only formal P1 SFT run")
    if cfg != training.get("config") or cfg != preflight.get("config") or cfg.get("epochs") != 2:
        raise ValueError("SFT config records must agree and specify two epochs")
    expected_config = input_lock.get("training_config_sha256")
    expected_train = input_lock.get("training_inputs_sha256", {}).get(arm)
    if (not expected_config or not expected_train or preflight.get("config_source_sha256") != expected_config
            or preflight["splits"]["train"]["source_sha256"] != expected_train):
        raise ValueError("Adapter training config or arm-specific data differ from the frozen comparison lock")
    if preflight["splits"]["train"]["used_rows"] < 512 or preflight["splits"]["validation"]["used_rows"] < 32:
        raise ValueError("Formal P1 SFT requires at least 512 training and 32 development rows")
    history = records["history"]
    if (history != training.get("history") or [row["epoch"] for row in history] != [1, 2]
            or training.get("optimizer_steps") != preflight["plan"]["optimizer_steps"]
            or training.get("nonzero_gradient_steps", 0) < 1):
        raise ValueError("SFT completion/history/optimizer evidence is inconsistent")
    if any(not math.isfinite(row["validation_answer_token_loss"]) for row in history):
        raise ValueError("SFT history contains a non-finite development loss")
    metrics = strict_json((adapter / "epoch_metrics.json").read_bytes())
    if metrics != history[epoch - 1]:
        raise ValueError("Epoch adapter metrics disagree with the completed training history")
    original = records["model_provenance"]
    if ((original.get("model"), original.get("revision")) != (MODEL_ID, MODEL_REVISION)
            or original.get("verified_files_sha256") != base_provenance["verified_files_sha256"]):
        raise ValueError("SFT adapter was not trained on this exact locked original base")
    adapter_cfg = strict_json((adapter / "adapter_config.json").read_bytes())
    if ((adapter_cfg.get("peft_type"), adapter_cfg.get("task_type"), adapter_cfg.get("bias")) != ("LORA", "CAUSAL_LM", "none")
            or adapter_cfg.get("r") != cfg["lora_rank"] or adapter_cfg.get("lora_alpha") != cfg["lora_alpha"]
            or adapter_cfg.get("lora_dropout") != cfg["lora_dropout"]
            or set(adapter_cfg.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj"}):
        raise ValueError("Saved adapter configuration differs from P1 training settings")
    inspected = inspect_adapter_weights(adapter / "adapter_model.safetensors")
    return {"weights_sha256": inspected["sha256"], "config_sha256": sha256_file(adapter / "adapter_config.json"),
            "epoch_metrics_sha256": sha256_file(adapter / "epoch_metrics.json"),
            "training_sha256": sha256_file(run / "training.json"),
            "training_config_sha256": expected_config, "training_data_sha256": expected_train}


def run_identity(arm, epoch, source_hashes, base_provenance, adapter_info):
    packages = {}
    for package in ("torch", "transformers", "peft", "safetensors"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"schema": "p1-evaluation-v1", "purpose": PURPOSE, "arm": arm, "epoch": epoch,
            **source_hashes, "base_model": MODEL_ID, "base_revision": MODEL_REVISION,
            "base_files_sha256": base_provenance["verified_files_sha256"], "adapter": adapter_info,
            "decode": DECODE, "prompt_version": PROMPT_VERSION,
            "prompt_code_sha256": sha256_file(Path(__file__).with_name("prompts.py")),
            "evaluator_sha256": sha256_file(__file__), "runtime_packages": packages,
            "support_code_sha256": {name: sha256_file(Path(__file__).with_name(name))
                                    for name in ("common.py", "p1_contract.py", "train_p1.py", "verify_p1.py")},
            "dtype": "bfloat16", "attention": "sdpa", "batch_size": 1}


def output_directory(path, bucket, resume):
    root = PROJECT_ROOT.resolve()
    parent = root / bucket
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = root / path
    if parent.resolve() != parent or path.is_symlink() or path.resolve().parent != parent:
        raise ValueError(f"Output must be a direct P1 directory under {bucket}, without symlinks")
    if resume:
        if not path.is_dir() or any(child.is_symlink() or not child.is_file() for child in path.iterdir()):
            raise ValueError("Resume requires an existing P1 directory containing only regular files")
    elif path.exists():
        raise FileExistsError("Existing outputs require explicit --resume")
    return path.resolve()


@contextmanager
def writer_lock(directory):
    with (directory / ".writer.lock").open("ab") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is already writing this P1 evaluation") from None
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def validate_record(record, previous, identity_sha, source_by_id):
    digest = record.get("record_sha256")
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    if (digest_json(body) != digest or record.get("previous_sha256") != previous
            or record.get("identity_sha256") != identity_sha):
        raise ValueError("Evaluation journal hash chain is invalid")
    source = source_by_id.get(record.get("id"))
    if (source is None or record.get("source_sha256") != source["source_sha256"]
            or record.get("report") != source["report"] or record.get("status") not in ("completed", "failed")):
        raise ValueError("Evaluation journal contains a changed/unknown source or nonterminal row")
    generation = record.get("generation", {})
    if (not isinstance(record.get("prediction"), str)
            or any(type(generation.get(key)) is not int or generation[key] < 0 for key in ("input_tokens", "output_tokens"))
            or type(generation.get("eos_completed")) is not bool
            or not isinstance(generation.get("elapsed_seconds"), (int, float))
            or not math.isfinite(generation["elapsed_seconds"]) or generation["elapsed_seconds"] < 0
            or not isinstance(record.get("token_ids"), list)
            or any(type(value) is not int for value in record["token_ids"])
            or len(record["token_ids"]) != generation["output_tokens"]):
        raise ValueError("Evaluation journal has invalid output/token/timing fields")
    if record["status"] == "completed":
        if not record["prediction"].strip() or not generation["eos_completed"] or record.get("error_code") is not None:
            raise ValueError("Completed journal rows must be nonempty EOS completions")
    elif not isinstance(record.get("error_code"), str) or not record["error_code"]:
        raise ValueError("Failed journal rows must preserve an explicit error code")
    return digest


def read_journal(path, rows, identity_sha, recover=False):
    """Recover only a trailing interrupted write; all complete records are immutable."""
    source_by_id = {row["id"]: row for row in rows}
    records, seen, previous, complete_bytes = [], set(), identity_sha, 0
    chunks = path.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(chunks):
        terminal = line.endswith(b"\n")
        try:
            record = strict_json(line)
        except (ValueError, UnicodeDecodeError):
            if terminal or index != len(chunks) - 1 or not recover:
                raise ValueError("Malformed complete journal record or recovery not authorized") from None
            backup = path.with_name("recovery-tail-" + hashlib.sha256(line).hexdigest() + ".bin")
            if backup.exists():
                if backup.read_bytes() != line:
                    raise ValueError("Recovery backup differs from the interrupted tail")
            else:
                with backup.open("xb") as stream:
                    stream.write(line)
                    stream.flush()
                    os.fsync(stream.fileno())
            with path.open("r+b") as stream:
                stream.truncate(complete_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            break
        previous = validate_record(record, previous, identity_sha, source_by_id)
        if record["id"] in seen:
            raise ValueError("Duplicate terminal source ID in the evaluation journal")
        seen.add(record["id"])
        records.append(record)
        complete_bytes += len(line)
        if not terminal:
            if not recover:
                raise ValueError("Journal requires explicit recovery of a missing final newline")
            with path.open("ab") as stream:
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
    return records


def append_result(path, result, records, identity_sha):
    if any(record["id"] == result["id"] for record in records):
        raise ValueError("Refusing to rerun or append an already terminal source ID")
    record = dict(result, identity_sha256=identity_sha,
                  previous_sha256=records[-1]["record_sha256"] if records else identity_sha)
    record["record_sha256"] = digest_json(record)
    line = (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
    with path.open("ab") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    records.append(record)


def encode_prompt(tokenizer, report):
    ids = tokenizer.apply_chat_template(messages(report), tokenize=True, add_generation_prompt=True, return_dict=False)
    if not isinstance(ids, list) or not ids or any(type(value) is not int for value in ids):
        raise ValueError("Chat template must produce a nonempty flat list of token IDs")
    return ids


def pending_rows(rows, records):
    terminal = {record["id"] for record in records}
    return [row for row in rows if row["id"] not in terminal]


def generate_one(model, tokenizer, prompt, torch, generation_config):
    ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=generation_config)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    values = output[0].tolist()
    if values[:len(prompt)] != prompt:
        raise ValueError("Generated output does not preserve the exact prompt")
    decoded = generation_result(tokenizer, values[len(prompt):])
    code = None if decoded["eos_completed"] and decoded["nonempty"] else (
        "generation_limit_without_eos" if not decoded["eos_completed"] else "empty_output")
    return {"status": "failed" if code else "completed", "prediction": decoded["text"],
            "error_code": code, "token_ids": decoded["token_ids"],
            "generation": {"input_tokens": len(prompt), "output_tokens": len(decoded["token_ids"]),
                           "eos_completed": decoded["eos_completed"], "elapsed_seconds": elapsed}}


def load_runtime(model_path, adapter_path, torch):
    from transformers import AutoModelForCausalLM, GenerationConfig
    tokenizer = load_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(str(Path(model_path).resolve()), local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="sdpa", use_safetensors=True).to("cuda")
    if adapter_path is not None:
        from peft import PeftModel, get_peft_model_state_dict
        from safetensors.torch import load_file
        model = PeftModel.from_pretrained(model, str(Path(adapter_path).resolve()), is_trainable=False,
                                         local_files_only=True, autocast_adapter_dtype=True)
        model.set_adapter("default")
        saved = load_file(str(Path(adapter_path) / "adapter_model.safetensors"), device="cpu")
        loaded = get_peft_model_state_dict(model, adapter_name="default")
        same_tensor_structure(saved, loaded)
        for key, value in saved.items():
            actual = loaded[key].detach().cpu()
            if not bool(torch.isfinite(actual).all()) or not torch.equal(actual, value.to(dtype=actual.dtype)):
                raise ValueError("Reloaded adapter values differ from the saved checkpoint")
        layers = [module for module in model.modules() if hasattr(module, "lora_A") and "default" in module.lora_A]
        if not layers or any(module.disable_adapters or list(module.active_adapters) != ["default"] for module in layers):
            raise ValueError("Requested SFT adapter is not active")
    elif getattr(model, "peft_config", None):
        raise ValueError("The base arm unexpectedly contains an adapter")
    model.config.use_cache = True
    model.eval()
    config = GenerationConfig(max_new_tokens=512, do_sample=False, num_beams=1, use_cache=True,
                              eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                              bos_token_id=tokenizer.bos_token_id)
    return model, tokenizer, config


def public_manifest(identity, identity_sha, rows, records, journal_path, attempt, attempt_hash):
    failed = sum(record["status"] == "failed" for record in records)
    return {"schema": "p1-evaluation-manifest-v1", "purpose": PURPOSE, "arm": identity["arm"], "epoch": identity["epoch"],
            "status": ("interrupted" if len(records) < len(rows) else "completed_with_failures" if failed else "completed"),
            "quality_correctness_claim": False,
            "hashes": {"identity": identity_sha, "input_lock": identity["input_lock_sha256"],
                       "sources": identity["sources_sha256"], "journal": sha256_file(journal_path), "attempt": attempt_hash},
            "counts": {"requested": len(rows), "terminal": len(records), "completed": len(records) - failed,
                       "failed": failed, "pending": len(rows) - len(records),
                       "eos_completed": sum(record["generation"]["eos_completed"] for record in records)},
            "tokens": {key: sum(record["generation"][key] for record in records) for key in ("input_tokens", "output_tokens")},
            "elapsed_seconds": {"generation_all_terminal_rows": sum(record["generation"]["elapsed_seconds"] for record in records),
                                "current_attempt": attempt["elapsed_seconds"], "current_model_load": attempt["model_load_seconds"]},
            "memory_bytes": attempt["memory_bytes"]}


def evaluate(arm, sources, input_lock, model_path, output, public_output, adapter=None, epoch=None, resume=False, base_lock=None):
    rows, lock, source_hashes = load_sources(sources, input_lock)
    if arm == "base" and (adapter is not None or epoch is not None):
        raise ValueError("The base arm must not use an adapter or epoch")
    raw = output_directory(output, "data/p1", resume)
    public = output_directory(public_output, "results/p1", resume)
    provenance = verify_original_snapshot(model_path, require_weights=True, base_lock=base_lock)
    adapter_info = adapter_identity(arm, adapter, epoch, lock, provenance)
    identity = run_identity(arm, epoch, source_hashes, provenance, adapter_info)
    identity_sha = digest_json(identity)
    if not resume:
        raw.mkdir(parents=True, exist_ok=False)
        raw.chmod(0o700)
        public.mkdir(parents=True, exist_ok=False)
    with writer_lock(raw):
        frozen_path, journal = raw / "run.lock.json", raw / "predictions.jsonl"
        binding = digest_json({"raw_directory": raw.name, "public_directory": public.name})
        frozen = {"identity": identity, "identity_sha256": identity_sha, "output_binding_sha256": binding}
        public_lock = {"identity_sha256": identity_sha, "output_binding_sha256": binding}
        if resume:
            stored = strict_json(frozen_path.read_bytes())
            if stored != frozen or strict_json((public / "run.lock.json").read_bytes()) != public_lock:
                raise ValueError("Resume input/model/adapter/epoch/code/runtime identity has changed")
            records = read_journal(journal, rows, identity_sha, recover=True)
        else:
            write_new_json(frozen_path, frozen)
            write_new_json(public / "run.lock.json", public_lock)
            journal.touch(exist_ok=False)
            journal.chmod(0o600)
            records = []
        number = 1
        while (raw / f"attempt-{number:06}.json").exists() or (public / f"manifest-{number:06}.json").exists():
            number += 1
        started = time.perf_counter()
        attempt = {"elapsed_seconds": 0.0, "model_load_seconds": 0.0, "failure": None,
                   "memory_bytes": {"peak_allocated": None, "peak_reserved": None}}
        torch = None
        try:
            pending = pending_rows(rows, records)
            if pending:
                torch = require_cuda()
                loading = time.perf_counter()
                model, tokenizer, generation_config = load_runtime(model_path, adapter, torch)
                torch.cuda.synchronize()
                attempt["model_load_seconds"] = time.perf_counter() - loading
                torch.cuda.reset_peak_memory_stats()
                for row in pending:
                    result = dict(row, status="failed", prediction="", error_code=None, token_ids=[],
                                  generation={"input_tokens": 0, "output_tokens": 0, "eos_completed": False, "elapsed_seconds": 0.0})
                    row_started = time.perf_counter()
                    try:
                        prompt = encode_prompt(tokenizer, row["report"])
                        result["generation"]["input_tokens"] = len(prompt)
                        if len(prompt) > DECODE["max_input_tokens"]:
                            result["error_code"] = "input_overlength_no_truncation"
                        else:
                            result.update(generate_one(model, tokenizer, prompt, torch, generation_config))
                    except Exception:
                        result["error_code"] = "generation_exception"
                        result["generation"]["elapsed_seconds"] = time.perf_counter() - row_started
                        result["private_error"] = traceback.format_exc()
                        append_result(journal, result, records, identity_sha)
                        raise
                    append_result(journal, result, records, identity_sha)
        except Exception:
            attempt["failure"] = traceback.format_exc()
        finally:
            attempt["elapsed_seconds"] = time.perf_counter() - started
            if torch is not None:
                try:
                    attempt["memory_bytes"] = {"peak_allocated": torch.cuda.max_memory_allocated(),
                                               "peak_reserved": torch.cuda.max_memory_reserved()}
                except Exception:
                    attempt["memory_query_failure"] = traceback.format_exc()
            attempt_path = raw / f"attempt-{number:06}.json"
            write_new_json(attempt_path, attempt)
            summary = public_manifest(identity, identity_sha, rows, records, journal, attempt, sha256_file(attempt_path))
            write_new_json(public / f"manifest-{number:06}.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("base", "S0", "S1"))
    parser.add_argument("--sources", required=True)
    parser.add_argument("--input-lock", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--epoch", type=int, choices=(1, 2))
    parser.add_argument("--out", required=True, help="New ignored raw directory directly under data/p1")
    parser.add_argument("--public-out", required=True, help="New public directory directly under results/p1")
    parser.add_argument("--resume", action="store_true", help="Verify the exact run identity; process only IDs without terminal records")
    parser.add_argument("--base-lock")
    args = parser.parse_args(argv)
    try:
        summary = evaluate(args.arm, args.sources, args.input_lock, args.model, args.out, args.public_out,
                           args.adapter, args.epoch, args.resume, args.base_lock)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as error:
        print(json.dumps({"status": "rejected", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
    if summary["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
