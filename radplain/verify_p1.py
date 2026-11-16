"""Reload a saved P1 pilot adapter and probe ten already-used development rows.

This is an engineering check, not a quality evaluation or checkpoint selection.
Raw text stays in ignored data/p1; the public result contains hashes, timing and
completion/check counts only. No model library is imported until verify().
"""
import argparse
import gc
import hashlib
import json
import math
import re
import struct
import time
import traceback
from pathlib import Path

from .common import fingerprint, read_jsonl
from .p1_contract import MODEL_ID, MODEL_REVISION, PROJECT_ROOT, safe_output_path, sha256_file, verify_original_snapshot
from .prompts import messages
from .train_p1 import load_tokenizer, require_cuda, write_new_json

PROBE_COUNT = 10
PLAN_PATH = PROJECT_ROOT / "examples/p1/pilot_generation_plan_20261117.json"
ARMS = ("original_base", "saved_reloaded_pilot_adapter")
FLOAT_BITS = {"F16": ("H", 2, 0x7C00), "BF16": ("H", 2, 0x7F80),
              "F32": ("I", 4, 0x7F800000), "F64": ("Q", 8, 0x7FF0000000000000)}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def raw_output_path(path):
    bucket = PROJECT_ROOT / "data/p1"
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if bucket.resolve() != bucket or path.is_symlink():
        raise ValueError("Raw P1 output must not use symlinks")
    if path.exists():
        raise FileExistsError("Refusing an existing raw P1 output directory")
    if path.resolve().parent != bucket:
        raise ValueError("Raw probe output must be a new direct directory under ignored data/p1")
    return path.resolve()


def completed_pilot(run, data):
    run = Path(run).resolve(strict=True)
    loaded = {name: json.loads((run / f"{name}.json").read_text(encoding="utf-8"))
              for name in ("config", "preflight", "training", "history", "selection", "model_provenance")}
    training, preflight, selection = (loaded[key] for key in ("training", "preflight", "selection"))
    if any(value.get("mode") != "pilot" for value in (training, preflight, selection)):
        raise ValueError("Reload verification accepts a completed engineering pilot only")
    if training.get("training_completed") is not True or (run / "failure.json").exists():
        raise ValueError("Pilot training is incomplete or has a failure record")
    if any(value.get("final_test_read") is not False for value in (training, preflight, selection)):
        raise ValueError("Expected a development-only pilot")
    if training.get("optimizer_steps") != preflight["plan"]["optimizer_steps"]:
        raise ValueError("Pilot optimizer step count does not match its preflight")
    if training.get("config") != loaded["config"] or preflight.get("config") != loaded["config"]:
        raise ValueError("Pilot config records disagree")
    if training.get("history") != loaded["history"] or not loaded["history"]:
        raise ValueError("Pilot history records disagree or are empty")
    best = min(loaded["history"], key=lambda value: value["validation_answer_token_loss"])
    epoch = selection.get("epoch")
    if (type(epoch) is not int or epoch != best["epoch"]
            or selection.get("validation_answer_token_loss") != best["validation_answer_token_loss"]):
        raise ValueError("Saved selection does not match the development-loss choice")
    if selection.get("adapter_path") != f"epoch-{epoch}":
        raise ValueError("Selection must point to its exact epoch directory")
    adapter = run / selection["adapter_path"]
    if adapter.is_symlink() or adapter.resolve(strict=True).parent != run:
        raise ValueError("Selected adapter must remain inside its pilot run")
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if (config.get("peft_type"), config.get("task_type"), config.get("bias")) != ("LORA", "CAUSAL_LM", "none"):
        raise ValueError("Expected the saved P1 causal-LM LoRA adapter")
    if (config.get("r") != loaded["config"]["lora_rank"]
            or config.get("lora_alpha") != loaded["config"]["lora_alpha"]
            or set(config.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj"}):
        raise ValueError("Saved adapter configuration does not match the pilot")
    validation = Path(data).resolve(strict=True) / "validation.zh.jsonl"
    development = preflight["splits"]["validation"]
    if sha256_file(validation) != development["source_sha256"]:
        raise ValueError("Development file differs from the one used by the pilot")
    rows = read_jsonl(validation)
    selected = development["selected_rows"]
    if len(selected) < PROBE_COUNT:
        raise ValueError("The pilot must contain at least ten development rows")
    indices = [value["row"] for value in selected]
    if any(type(value) is not int for value in indices) or indices != sorted(set(indices)):
        raise ValueError("Development selection must be unique and in source-file order")
    probes = []
    for record in selected[:PROBE_COUNT]:
        if not 1 <= record["row"] <= len(rows):
            raise ValueError("Development row is outside its source file")
        row = rows[record["row"] - 1]
        if (row.get("teacher_accepted") is not True or row.get("id") != record["id"]
                or fingerprint(row["report"]) != record["report_sha256"]):
            raise ValueError("Development row does not match the saved selection")
        probes.append(row)
    hashes = {name: sha256_file(run / f"{name}.json") for name in loaded}
    hashes.update({"development_file": sha256_file(validation),
                   "development_selection": digest_json(selected[:PROBE_COUNT]),
                   "adapter_config": sha256_file(adapter / "adapter_config.json")})
    return adapter, probes, loaded, hashes


def validate_generation_plan(plan, rows, run_records):
    if (plan.get("purpose") != "engineering_save_reload_and_generation_smoke_only"
            or plan.get("source") != "old_development_only_not_final_test"
            or plan.get("quality_claim") is not False or plan.get("model_predictions_seen") is not False
            or plan.get("arms") != list(ARMS)
            or plan.get("decode") != {"do_sample": False, "max_new_tokens": 256}):
        raise ValueError("The fixed development-only generation plan is required")
    preflight = run_records["preflight"]
    validation = preflight["splits"]["validation"]
    if (validation["source_sha256"] != plan.get("validation_file_sha256")
            or preflight["config_source_sha256"] != plan.get("training_config_sha256")
            or preflight["splits"]["train"]["source_sha256"] != plan.get("training_data_sha256")):
        raise ValueError("Pilot input hashes differ from the frozen generation plan")
    if [value["row"] for value in validation["selected_rows"][:PROBE_COUNT]] != list(range(1, PROBE_COUNT + 1)):
        raise ValueError("The frozen probe requires the first ten validation rows in source order")
    actual = [{"id": row.get("id"), "source_sha256": hashlib.sha256(row["report"].encode()).hexdigest()}
              for row in rows]
    if len(rows) != PROBE_COUNT or actual != plan.get("reports"):
        raise ValueError("Development probes differ from the frozen report identities")


def inspect_adapter_weights(path):
    """Read IEEE float bits on CPU, including BF16, without importing torch.

    PEFT also validates the safetensors file when it independently reloads it.
    This checks every stored value rather than just the file size or header.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size < 10:
        raise ValueError("Saved adapter weights must be a nonempty regular file")
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= header_size <= min(10 * 1024 * 1024, path.stat().st_size - 8):
            raise ValueError("Invalid safetensors header size")
        metadata = json.loads(stream.read(header_size))
        tensors = {key: value for key, value in metadata.items() if key != "__metadata__"}
        if not tensors:
            raise ValueError("Saved adapter contains no tensors")
        shapes, regions, pairs = {}, [], {}
        for name, info in tensors.items():
            match = re.fullmatch(r"(.+\.(?:q_proj|k_proj|v_proj|o_proj))\.lora_([AB])\.weight", name)
            if match is None or info["dtype"] not in FLOAT_BITS:
                raise ValueError("Unexpected adapter tensor name or non-floating dtype")
            shape = info["shape"]
            if len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
                raise ValueError("LoRA tensors must be nonempty matrices")
            code, width, mask = FLOAT_BITS[info["dtype"]]
            begin, end = info["data_offsets"]
            if (type(begin) is not int or type(end) is not int or begin < 0
                    or end - begin != math.prod(shape) * width
                    or end > path.stat().st_size - 8 - header_size):
                raise ValueError("Invalid adapter tensor byte range")
            stream.seek(8 + header_size + begin)
            remaining = end - begin
            while remaining:
                content = stream.read(min(1024 * 1024, remaining))
                if not content or any(bits[0] & mask == mask for bits in struct.iter_unpack("<" + code, content)):
                    raise ValueError("Saved adapter contains a non-finite or truncated tensor")
                remaining -= len(content)
            shapes[name] = shape
            regions.append((begin, end))
            pairs.setdefault(match[1], set()).add(match[2])
        cursor = 0
        for begin, end in sorted(regions):
            if begin != cursor:
                raise ValueError("Adapter tensor ranges overlap or leave a gap")
            cursor = end
        if cursor != path.stat().st_size - 8 - header_size or any(value != {"A", "B"} for value in pairs.values()):
            raise ValueError("Adapter tensor payload or LoRA pairs are incomplete")
    return {"sha256": sha256_file(path), "shapes": shapes}


def generation_result(tokenizer, suffix):
    eos = tokenizer.eos_token_id
    completed = eos in suffix
    ending = suffix.index(eos) + 1 if completed else len(suffix)
    ids = suffix[:ending]
    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
    return {"text": text, "token_ids": ids, "eos_completed": completed,
            "nonempty": bool(text), "contains_han": bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))}


def same_tensor_structure(saved, loaded):
    if set(saved) != set(loaded) or any(list(saved[key].shape) != list(loaded[key].shape) for key in saved):
        raise ValueError("Reloaded adapter tensor keys or shapes differ from the saved checkpoint")


def generate_rows(model, tokenizer, prompts, rows, decode_config, torch):
    results = []
    for prompt, row in zip(prompts, rows):
        ids = torch.tensor([prompt], dtype=torch.long, device="cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=decode_config)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        values = output[0].tolist()
        if values[:len(prompt)] != prompt:
            raise ValueError("Generated sequence does not retain its exact prompt")
        result = generation_result(tokenizer, values[len(prompt):])
        results.append({"id": row.get("id"), "report": row["report"], "reference_zh": row["reference_zh"],
                        "generation_seconds": elapsed, **result})
    return results


def completion_summary(rows):
    return {"requested": PROBE_COUNT, "generated": len(rows),
            **{key: sum(row[key] for row in rows) for key in ("eos_completed", "nonempty", "contains_han")}}


def verify(run, data, model_path, output, raw_output, base_lock=None, plan_path=None):
    output = safe_output_path(output, "preflight")
    raw_output = raw_output_path(raw_output)
    torch = require_cuda()
    output.mkdir(parents=True, exist_ok=False)
    raw_output.mkdir(parents=True, exist_ok=False)
    raw_output.chmod(0o700)
    started = time.perf_counter()
    summary = {"schema": "p1-reload-check-v1", "status": "failed", "checks": {
        "engineering_only": True, "final_test_read": False, "formal_effectiveness_claim": False,
        "adapter_nonempty_finite": False, "adapter_reloaded_exactly": False},
        "hashes": {}, "timing_seconds": {}, "completion": {}}
    try:
        adapter, rows, run_records, hashes = completed_pilot(run, data)
        plan_path = Path(plan_path or PLAN_PATH)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        validate_generation_plan(plan, rows, run_records)
        hashes["generation_plan"] = sha256_file(plan_path)
        weight_info = inspect_adapter_weights(adapter / "adapter_model.safetensors")
        hashes["adapter_weights"] = weight_info["sha256"]
        summary["hashes"] = hashes
        summary["checks"]["adapter_nonempty_finite"] = True
        provenance = verify_original_snapshot(model_path, require_weights=True, base_lock=base_lock)
        saved_base = run_records["model_provenance"]
        if (saved_base.get("model"), saved_base.get("revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("Pilot provenance has an unexpected original model revision")
        if saved_base.get("verified_files_sha256") != provenance["verified_files_sha256"]:
            raise ValueError("Reload base files differ from those used for pilot training")
        hashes["original_base_lock"] = provenance["upstream_lock_sha256"]
        tokenizer = load_tokenizer(model_path)
        prompts = [tokenizer.apply_chat_template(messages(row["report"]), tokenize=True,
                   add_generation_prompt=True, return_dict=False) for row in rows]
        if any(not isinstance(prompt, list) or not prompt or any(type(token) is not int for token in prompt)
               or len(prompt) > run_records["config"]["max_sequence_length"] for prompt in prompts):
            raise ValueError("Invalid or overlength development prompt; truncation is forbidden")
        from peft import PeftModel, get_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, GenerationConfig
        decode_config = GenerationConfig(max_new_tokens=plan["decode"]["max_new_tokens"], do_sample=False, num_beams=1,
                                         use_cache=True, eos_token_id=tokenizer.eos_token_id,
                                         pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id)
        hashes["decoding_config"] = digest_json(decode_config.to_dict())
        write_new_json(raw_output / "manifest.json", {"decode_config": decode_config.to_dict(),
                       "selection": run_records["preflight"]["splits"]["validation"]["selected_rows"][:PROBE_COUNT],
                       "selection_rule": "first ten already-used development rows in source-file order"})
        for arm in ARMS:
            torch.cuda.synchronize()
            load_started = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(str(Path(model_path).resolve()), local_files_only=True,
                        trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="sdpa", use_safetensors=True).to("cuda")
            if arm == "saved_reloaded_pilot_adapter":
                model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False,
                                                  local_files_only=True, autocast_adapter_dtype=True)
                model.set_adapter("default")
                saved_state = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
                loaded_state = get_peft_model_state_dict(model, adapter_name="default")
                same_tensor_structure(saved_state, loaded_state)
                for key, value in saved_state.items():
                    actual = loaded_state[key].detach().cpu()
                    if not bool(torch.isfinite(actual).all()) or not torch.equal(actual, value.to(dtype=actual.dtype)):
                        raise ValueError("Reloaded adapter values differ from the saved checkpoint")
                layers = [module for module in model.modules() if hasattr(module, "lora_A") and "default" in module.lora_A]
                if not layers or any(module.disable_adapters or list(module.active_adapters) != ["default"] for module in layers):
                    raise ValueError("The reloaded adapter is not active")
                del saved_state, loaded_state, layers
                summary["checks"]["adapter_reloaded_exactly"] = True
            model.config.use_cache = True
            model.eval()
            torch.cuda.synchronize()
            summary["timing_seconds"][arm + "_load"] = time.perf_counter() - load_started
            results = generate_rows(model, tokenizer, prompts, rows, decode_config, torch)
            raw_file = raw_output / f"{arm}.jsonl"
            with raw_file.open("x", encoding="utf-8") as stream:
                for result in results:
                    stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            raw_file.chmod(0o600)
            hashes[arm + "_raw"] = sha256_file(raw_file)
            summary["completion"][arm] = completion_summary(results)
            summary["timing_seconds"][arm + "_generation"] = sum(row["generation_seconds"] for row in results)
            del model, results
            gc.collect()
            torch.cuda.empty_cache()
        if all(value == PROBE_COUNT for counts in summary["completion"].values() for value in counts.values()):
            summary["status"] = "passed"
    except Exception:
        write_new_json(raw_output / "failure.json", {"traceback": traceback.format_exc()})
        summary["hashes"]["private_failure"] = sha256_file(raw_output / "failure.json")
    summary["timing_seconds"]["total"] = time.perf_counter() - started
    write_new_json(output / "verification.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Completed pilot run containing selection.json")
    parser.add_argument("--data", required=True, help="Same data directory used by that pilot; only validation is read")
    parser.add_argument("--model", required=True, help="Locked original local base snapshot")
    parser.add_argument("--out", required=True, help="New public directory under results/p1")
    parser.add_argument("--raw-out", required=True, help="New ignored directory under data/p1")
    parser.add_argument("--base-lock")
    parser.add_argument("--plan", help="Frozen development generation plan; defaults to examples/p1/pilot_generation_plan_20261117.json")
    args = parser.parse_args(argv)
    try:
        result = verify(args.run, args.data, args.model, args.out, args.raw_out, args.base_lock, args.plan)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        # Raw diagnostics may contain report text or local paths; do not print them.
        print(json.dumps({"status": "failed_before_verification", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
