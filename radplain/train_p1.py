"""Isolated P1 CPU preflight and scheduled-GPU LoRA entry point.

The minimal training loop follows radplain.train's LoRA, dev-loss selection and
padding design. It does not modify or invoke V1/V2/V3 training or final tests.
Torch, PEFT and model-loading imports are confined to the explicit train action.
"""
import argparse
import itertools
import json
import math
import os
import platform
import time
from pathlib import Path

from .common import load_config
from .p1_contract import (
    collate_lists, prepare_data, safe_output_path, sha256_file, validate_config,
    verify_original_snapshot,
)


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(path).resolve()), local_files_only=True, trust_remote_code=False,
        padding_side="right",
    )
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise ValueError("A native chat template and EOS token are required")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare(config_path, data_dir, tokenizer_path, mode, base_lock=None):
    config = load_config(config_path)
    validate_config(config, mode)
    provenance = verify_original_snapshot(tokenizer_path, base_lock=base_lock)
    tokenizer = load_tokenizer(tokenizer_path)
    encoded, report = prepare_data(tokenizer, config, data_dir, mode)
    report.update({"config": config, "config_source_sha256": sha256_file(config_path),
                   "tokenizer_provenance": provenance, "python": platform.python_version()})
    return config, tokenizer, encoded, report


def preflight(config_path, data_dir, tokenizer_path, output, mode, base_lock=None):
    output = safe_output_path(output, "preflight")
    config, _, _, report = prepare(config_path, data_dir, tokenizer_path, mode, base_lock)
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "config.json", config)
    write_new_json(output / "preflight.json", report)
    return report


def require_cuda():
    # This import must remain outside all CPU preflight and module import paths.
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Training requires torch in a scheduled CUDA environment") from error
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a scheduled CUDA allocation; CPU/MPS training is disabled")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("P1 requires a CUDA GPU with bfloat16 support (planned: A100)")
    return torch


def train(config_path, data_dir, tokenizer_path, model_path, output, mode, base_lock=None):
    output = safe_output_path(output, "train")
    config = load_config(config_path)
    validate_config(config, mode)
    torch = require_cuda()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This P1 pilot supports one process and one GPU only")
    model_provenance = verify_original_snapshot(model_path, require_weights=True, base_lock=base_lock)
    config, tokenizer, data, report = prepare(config_path, data_dir, tokenizer_path, mode, base_lock)
    token_files = report["tokenizer_provenance"]["verified_files_sha256"]
    model_files = model_provenance["verified_files_sha256"]
    if any(model_files.get(name) != digest for name, digest in token_files.items()):
        raise ValueError("Training tokenizer files do not match the original model snapshot")

    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
    from .common import seed_everything

    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "config.json", config)
    write_new_json(output / "preflight.json", report)
    write_new_json(output / "model_provenance.json", model_provenance)
    try:
        seed_everything(config["seed"])
        model = AutoModelForCausalLM.from_pretrained(
            str(Path(model_path).resolve()), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", use_safetensors=True,
        ).to("cuda")
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            r=config["lora_rank"], lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"], target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none", task_type="CAUSAL_LM",
        ))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

        def collator(rows):
            return {key: torch.tensor(value, dtype=torch.long)
                    for key, value in collate_lists(rows, tokenizer.pad_token_id).items()}

        loader = DataLoader(data["train"], batch_size=config["train_batch_size"], shuffle=True,
                            generator=torch.Generator().manual_seed(config["seed"]),
                            collate_fn=collator, num_workers=0)
        dev = DataLoader(data["validation"], batch_size=config["train_batch_size"],
                         collate_fn=collator, num_workers=0)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.01)
        total_steps = report["plan"]["optimizer_steps"]
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.06), total_steps)
        accumulation = config["gradient_accumulation_steps"]
        start, history, best, best_epoch, update = time.perf_counter(), [], math.inf, None, 0
        gradient_norms = []
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            weighted_loss, tokens = 0.0, 0
            batches = iter(loader)
            while group := list(itertools.islice(batches, accumulation)):
                # Each optimizer update is a token mean, including a partial final
                # group; differently sized answers are not equally weighted batches.
                counts = [int((batch["labels"][:, 1:] != -100).sum()) for batch in group]
                group_tokens = sum(counts)
                if group_tokens == 0:
                    raise ValueError("Accumulation group has no supervised tokens")
                optimizer.zero_grad(set_to_none=True)
                for batch, count in zip(group, counts):
                    batch = {key: value.to("cuda") for key, value in batch.items()}
                    loss = model(**batch).loss
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Non-finite training loss")
                    weighted_loss += float(loss.item()) * count
                    tokens += count
                    (loss * (count / group_tokens)).backward()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
                gradient_norms.append(gradient_norm)
                optimizer.step()
                scheduler.step()
                if not all(bool(torch.isfinite(parameter).all()) for parameter in parameters):
                    raise FloatingPointError("Non-finite trainable parameters after optimizer update")
                update += 1
                print(json.dumps({"epoch": epoch, "optimizer_step": update,
                                  "planned_steps": total_steps,
                                  "gradient_norm_before_clipping": gradient_norm,
                                  "train_answer_token_loss": weighted_loss / tokens}), flush=True)
            model.eval()
            dev_loss, dev_tokens = 0.0, 0
            with torch.inference_mode():
                for batch in dev:
                    batch = {key: value.to("cuda") for key, value in batch.items()}
                    count = int((batch["labels"][:, 1:] != -100).sum())
                    loss = model(**batch).loss
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Non-finite validation loss")
                    dev_loss += float(loss.item()) * count
                    dev_tokens += count
            if not dev_tokens:
                raise ValueError("Validation has no supervised tokens")
            score = dev_loss / dev_tokens
            checkpoint = output / f"epoch-{epoch}"
            checkpoint.mkdir(exist_ok=False)
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            record = {"epoch": epoch, "train_answer_token_loss": weighted_loss / tokens,
                      "validation_answer_token_loss": score, "optimizer_steps": update,
                      "elapsed_seconds": time.perf_counter() - start}
            history.append(record)
            write_new_json(checkpoint / "epoch_metrics.json", record)
            if score < best:
                best, best_epoch = score, epoch
        if update != total_steps:
            raise RuntimeError(f"Optimizer step count mismatch: {update} != {total_steps}")
        if not any(value > 0 for value in gradient_norms):
            raise RuntimeError("No nonzero trainable gradient was observed")
        torch.cuda.synchronize()
        write_new_json(output / "history.json", history)
        write_new_json(output / "selection.json", {
            "epoch": best_epoch, "adapter_path": f"epoch-{best_epoch}",
            "validation_answer_token_loss": best,
            "criterion": "lowest validation answer/EOS-token cross-entropy",
            "final_test_read": False, "mode": mode,
            "formal_effectiveness_claim": False,
        })
        summary = {"mode": mode, "purpose": report["purpose"], "config": config,
                   "history": history, "optimizer_steps": update, "final_test_read": False,
                   "training_completed": True, "formal_effectiveness_claim": False,
                   "gradient_norms_before_clipping": gradient_norms,
                   "nonzero_gradient_steps": sum(value > 0 for value in gradient_norms),
                   "elapsed_seconds": time.perf_counter() - start,
                   "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "trainable_parameters": sum(parameter.numel() for parameter in parameters),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "cuda": torch.version.cuda, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        write_new_json(output / "training.json", summary)
        return summary
    except Exception as error:
        write_new_json(output / "failure.json", {"type": type(error).__name__, "message": str(error),
                                                "training_completed": False, "final_test_read": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "train"),
                        help="preflight is a CPU-only dry run: no weights or GPU")
    parser.add_argument("--mode", required=True, choices=("pilot", "sft"),
                        help="pilot: 64–128 engineering samples; sft: at least 512 train / 32 dev")
    parser.add_argument("--config", default="configs/p1_pilot.json")
    parser.add_argument("--data", required=True, help="Directory with train.zh.jsonl and validation.zh.jsonl")
    parser.add_argument("--tokenizer", required=True, help="Local original tokenizer snapshot with P1 provenance lock")
    parser.add_argument("--model", help="Required for train: local original model snapshot with locked weights")
    parser.add_argument("--base-lock", help="Reviewed upstream Git/LFS manifest; default: configs/p1_base_files.lock.json")
    parser.add_argument("--out", required=True, help="New run under results/p1 (preflight) or artifacts/p1 (train)")
    args = parser.parse_args(argv)
    if args.action == "preflight":
        if args.model:
            parser.error("preflight accepts only --tokenizer; model weights are not used")
        report = preflight(args.config, args.data, args.tokenizer, args.out, args.mode, args.base_lock)
        print(json.dumps({"status": "cpu_preflight_passed", "mode": args.mode,
                          "train_rows": report["splits"]["train"]["used_rows"],
                          "validation_rows": report["splits"]["validation"]["used_rows"],
                          "plan": report["plan"], "out": str(Path(args.out).resolve()),
                          "gpu_training_verified": False}, ensure_ascii=False, indent=2))
    else:
        if not args.model:
            parser.error("train requires --model pointing to the locked original snapshot")
        train(args.config, args.data, args.tokenizer, args.model, args.out, args.mode, args.base_lock)


if __name__ == "__main__":
    main()
