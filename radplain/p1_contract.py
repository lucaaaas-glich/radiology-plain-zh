"""P1's CPU-only data, tokenization and provenance contract.

Prompt-prefix masking derives from radplain.train.encode_example. P1 additionally
removes template whitespace after EOS so only answer tokens and EOS are targets.
This module deliberately imports neither torch nor any model-loading library.
"""
import hashlib
import json
import math
import random
import re
from pathlib import Path

from .common import fingerprint, read_jsonl
from .prompts import PROMPT_VERSION, messages

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PROVENANCE_NAME = "p1_model_provenance.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_LOCK_PATH = PROJECT_ROOT / "configs/p1_base_files.lock.json"
TOKENIZER_FILES = {
    "config.json", "tokenizer.json", "tokenizer_config.json", "merges.txt",
    "vocab.json", "special_tokens_map.json", "added_tokens.json",
    "chat_template.jinja", "generation_config.json",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_original_snapshot(folder, *, require_weights=False, base_lock=None):
    """Verify a caller-prepared upstream lock; never download or load weights.

    Local provenance is bound to the repository's reviewed upstream Git/LFS
    manifest. Each required file is checked against both locks in a single read.
    CPU tokenization checks never read weights, even if weights are present.
    """
    folder = Path(folder).resolve(strict=True)
    if not folder.is_dir():
        raise ValueError("Snapshot must be a local directory")
    if any(re.search(r"^(merged|lora)([-_]|$)|[-_](merged|lora)$", part.lower()) for part in folder.parts):
        raise ValueError("Merged or LoRA paths cannot be used as the original base")
    if list(folder.glob("adapter*")) or list(folder.glob("*.bin")):
        raise ValueError("Adapter or pickle/bin model files are forbidden in a P1 base snapshot")
    lock_path = folder / PROVENANCE_NAME
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("model"), lock.get("revision"), lock.get("kind")) != (
        MODEL_ID, MODEL_REVISION, "original_base"
    ):
        raise ValueError("P1 requires the locked, unmerged original Qwen base revision")
    upstream_path = Path(base_lock or UPSTREAM_LOCK_PATH).resolve(strict=True)
    upstream_sha256 = sha256_file(upstream_path)
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    if (upstream.get("model"), upstream.get("revision"), upstream.get("kind")) != (
        MODEL_ID, MODEL_REVISION, "original_base"
    ):
        raise ValueError("Unexpected upstream base lock model or revision")
    if lock.get("upstream_lock_sha256") != upstream_sha256:
        raise ValueError("Local provenance is not bound to the supplied upstream base lock")
    expected_files = upstream.get("files", {})
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("Upstream base lock must contain pinned file metadata")
    declared = lock.get("files_sha256")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("Provenance must contain files_sha256")
    for name, digest in declared.items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Only flat, relative snapshot filenames are allowed")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid SHA256 for {name}")
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    required |= {name for name in TOKENIZER_FILES if (folder / name).exists()}
    if require_weights:
        weights = {path.name for path in folder.glob("*.safetensors")}
        if not weights:
            raise ValueError("Training requires local original safetensors weights")
        index = folder / "model.safetensors.index.json"
        if index.exists():
            required.add(index.name)
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            if not weight_map or set(weight_map.values()) != weights:
                raise ValueError("Safetensors index and local weight shards disagree")
        elif weights != {"model.safetensors"}:
            raise ValueError("Sharded weights require a complete safetensors index")
        required |= weights
    missing = required - declared.keys()
    if missing:
        raise ValueError(f"Provenance does not cover all required files: {sorted(missing)}")
    if required - expected_files.keys():
        raise ValueError("Snapshot contains required files absent from the upstream base lock")
    if require_weights and weights != {name for name in expected_files if name.endswith(".safetensors")}:
        raise ValueError("Local weight set differs from the pinned upstream original weights")
    checked = {}
    for name in sorted(required):
        path = folder / name
        expected = expected_files[name]
        if not path.is_file() or path.stat().st_size != expected.get("size"):
            raise ValueError(f"Missing or modified locked snapshot file: {name}")
        digest = hashlib.sha256()
        git_digest = None if expected.get("lfs_sha256") else hashlib.sha1()
        if git_digest is not None:
            git_digest.update(b"blob " + str(path.stat().st_size).encode() + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                if git_digest is not None:
                    git_digest.update(chunk)
        if digest.hexdigest() != declared[name]:
            raise ValueError(f"Missing or modified locked snapshot file: {name}")
        if ((git_digest is None and digest.hexdigest() != expected["lfs_sha256"])
                or (git_digest is not None and git_digest.hexdigest() != expected.get("git_blob_sha1"))):
            raise ValueError(f"Snapshot file differs from pinned upstream Git/LFS content: {name}")
        checked[name] = declared[name]
    model_config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if model_config.get("model_type") != "qwen2" or model_config.get("quantization_config"):
        raise ValueError("Expected the unquantized original Qwen2 model configuration")
    return {"model": MODEL_ID, "revision": MODEL_REVISION, "kind": "original_base",
            "path": str(folder), "lock_sha256": sha256_file(lock_path),
            "upstream_lock_path": str(upstream_path), "upstream_lock_sha256": upstream_sha256,
            "upstream_git_lfs_verified": True, "verified_files_sha256": checked,
            "weights_verified": require_weights}


def validate_config(config, mode):
    if mode not in ("pilot", "sft"):
        raise ValueError("Select an explicit pilot or sft mode")
    if config.get("student_model") != MODEL_ID or config.get("model_revision") != MODEL_REVISION:
        raise ValueError("Config must pin the original Qwen model and revision")
    for key in ("seed", "max_sequence_length", "epochs", "train_batch_size",
                "gradient_accumulation_steps", "lora_rank", "lora_alpha"):
        value = config.get(key)
        if type(value) is not int or value < (0 if key == "seed" else 1):
            raise ValueError(f"{key} must be a valid positive integer (seed may be zero)")
    for key, lower, upper in (("learning_rate", 0, 1), ("lora_dropout", 0, 1)):
        value = config.get(key)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or not lower <= value < upper
                or (key == "learning_rate" and value == 0)):
            raise ValueError(f"Invalid {key}")
    if mode == "pilot":
        size = config.get("pilot_train_size")
        dev = config.get("pilot_validation_size")
        if type(size) is not int or not 64 <= size <= 128:
            raise ValueError("Pilot train size must be between 64 and 128")
        if type(dev) is not int or dev < 32:
            raise ValueError("Pilot validation size must be at least 32")
    effective = config["train_batch_size"] * config["gradient_accumulation_steps"]
    expected = config.get("expected_effective_batch_size")
    if expected is not None and expected != effective:
        raise ValueError("Configured effective batch size disagrees with batch × accumulation")


def encode_example(tokenizer, row, max_length):
    """Encode without truncation; return None for an overlength sequence."""
    if not isinstance(row.get("report"), str) or not row["report"].strip():
        raise ValueError("Each accepted row needs a nonempty report")
    if not isinstance(row.get("reference_zh"), str) or not row["reference_zh"].strip():
        raise ValueError("Each accepted row needs a nonempty reference_zh")
    prompt = messages(row["report"])
    prefix = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
    full = tokenizer.apply_chat_template(
        prompt + [{"role": "assistant", "content": row["reference_zh"]}],
        tokenize=True, return_dict=False)
    if any(not isinstance(value, list) or any(type(token) is not int for token in value)
           for value in (prefix, full)):
        raise ValueError("Chat template must return a flat list of integer token IDs")
    if full[:len(prefix)] != prefix:
        raise ValueError("Chat template prefix mismatch; refusing an incorrect training mask")
    eos = tokenizer.eos_token_id
    suffix = full[len(prefix):]
    if eos is None or suffix.count(eos) != 1:
        raise ValueError("Answer must have exactly one EOS token")
    eos_index = len(prefix) + suffix.index(eos)
    if eos_index == len(prefix):
        raise ValueError("Example has no supervised answer tokens")
    trailing = full[eos_index + 1:]
    if trailing and tokenizer.decode(trailing, skip_special_tokens=False).strip():
        raise ValueError("Unexpected non-whitespace template content after answer EOS")
    full = full[:eos_index + 1]
    if len(full) > max_length:
        return None
    labels = [-100] * len(prefix) + full[len(prefix):]
    return {"input_ids": full, "labels": labels,
            "prompt_tokens": len(prefix), "answer_tokens": len(full) - len(prefix),
            "template_trailing_tokens_ignored": len(trailing)}


def collate_lists(rows, pad_id):
    """Tensor-free equivalent of V1's padding and prompt mask contract."""
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    size = (max(len(row["input_ids"]) for row in rows) + 7) // 8 * 8
    result = {"input_ids": [], "labels": [], "attention_mask": []}
    for row in rows:
        padding = size - len(row["input_ids"])
        result["input_ids"].append(row["input_ids"] + [pad_id] * padding)
        result["labels"].append(row["labels"] + [-100] * padding)
        result["attention_mask"].append([1] * len(row["input_ids"]) + [0] * padding)
    return result


def prepare_data(tokenizer, config, data_dir, mode):
    validate_config(config, mode)
    data_dir = Path(data_dir).resolve(strict=True)
    encoded, reports, fingerprints = {}, {}, {}
    for split in ("train", "validation"):
        path = data_dir / f"{split}.zh.jsonl"
        rows = read_jsonl(path)
        eligible, exclusions = [], []
        for line, row in enumerate(rows, 1):
            if type(row.get("teacher_accepted")) is not bool:
                raise ValueError(f"{split} row {line}: teacher_accepted must be boolean")
            if not row["teacher_accepted"]:
                exclusions.append({"row": line, "id": row.get("id"), "reason": "teacher_rejected"})
                continue
            key = fingerprint(row.get("report", ""))
            if key in fingerprints:
                raise ValueError(f"Duplicate accepted report in {fingerprints[key]} and {split} row {line}")
            fingerprints[key] = f"{split} row {line}"
            value = encode_example(tokenizer, row, config["max_sequence_length"])
            if value is None:
                exclusions.append({"row": line, "id": row.get("id"), "reason": "too_long_no_truncation"})
                continue
            eligible.append((line, row.get("id"), key, value))
        target = config[f"pilot_{split}_size"] if mode == "pilot" else len(eligible)
        minimum = (64 if mode == "pilot" else 512) if split == "train" else 32
        if len(eligible) < max(minimum, target):
            raise ValueError(f"Insufficient accepted, in-length {split} rows: {len(eligible)}; "
                             f"need {max(minimum, target)} for {mode}")
        if mode == "pilot" and len(eligible) > target:
            chosen = set(random.Random(config["seed"] + (0 if split == "train" else 1))
                         .sample(range(len(eligible)), target))
            selected = [item for index, item in enumerate(eligible) if index in chosen]
        else:
            selected = eligible
        encoded[split] = [item[3] for item in selected]
        lengths = [len(item[3]["input_ids"]) for item in selected]
        tokens = sum(sum(label != -100 for label in item[3]["labels"][1:]) for item in selected)
        reports[split] = {
            "source_file": str(path), "source_sha256": sha256_file(path), "raw_rows": len(rows),
            "eligible_rows": len(eligible), "used_rows": len(selected),
            "pilot_not_selected_rows": len(eligible) - len(selected), "exclusions": exclusions,
            "selected_rows": [{"row": item[0], "id": item[1], "report_sha256": item[2]}
                              for item in selected],
            "input_tokens": sum(lengths), "supervised_answer_eos_tokens": tokens,
            "min_sequence_tokens": min(lengths), "max_sequence_tokens": max(lengths),
            "mean_sequence_tokens": sum(lengths) / len(lengths),
            "template_trailing_tokens_ignored": sum(item[3]["template_trailing_tokens_ignored"]
                                                   for item in selected),
        }
    count = len(encoded["train"])
    batch = config["train_batch_size"]
    accum = config["gradient_accumulation_steps"]
    microbatches = math.ceil(count / batch)
    steps = math.ceil(microbatches / accum)
    planned_steps = steps * config["epochs"]
    expected = config.get("expected_optimizer_steps")
    if expected is not None and expected != planned_steps:
        raise ValueError(f"Expected {expected} optimizer steps but data/config require {planned_steps}")
    report = {
        "schema": "p1-preflight-v1", "mode": mode, "prompt_version": PROMPT_VERSION,
        "purpose": "engineering_pilot_not_formal_effectiveness" if mode == "pilot" else "formal_sft",
        "final_test_read": False, "gpu_training_verified": False,
        "quality_review_claim": "Tokenization and structure checks do not establish reference quality",
        "splits": reports,
        "plan": {"train_microbatches_per_epoch": microbatches, "optimizer_steps_per_epoch": steps,
                 "optimizer_steps": planned_steps, "epochs": config["epochs"],
                 "train_batch_size": batch, "gradient_accumulation_steps": accum,
                 "effective_batch_size": batch * accum,
                 "last_update_microbatches": microbatches - (steps - 1) * accum,
                 "last_update_examples": count - (steps - 1) * accum * batch,
                 "supervised_tokens_all_epochs": reports["train"]["supervised_answer_eos_tokens"]
                                                 * config["epochs"],
                 "loss_normalization": "answer/EOS-token mean over each accumulation group"},
    }
    return encoded, report


def safe_output_path(output, action, *, project_root=None):
    """Only new P1 run directories; resolved symlinks cannot escape the P1 root."""
    if action not in ("preflight", "train"):
        raise ValueError("Unknown P1 action")
    root = Path(project_root or PROJECT_ROOT).resolve()
    bucket = root / ("results" if action == "preflight" else "artifacts") / "p1"
    if bucket.resolve() != bucket:
        raise ValueError("P1 output root must not be redirected through symlinks")
    path = Path(output).expanduser()
    if not path.is_absolute():
        path = root / path
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite an existing run: {path}")
    resolved = path.resolve()
    if resolved.parent != bucket:
        raise ValueError(f"Output must be a new direct run directory below {bucket}")
    return resolved
