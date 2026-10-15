"""LoRA training with prompt-masked loss and dev-only checkpoint selection."""
import argparse
import math
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import get_linear_schedule_with_warmup
from .common import load_config, read_jsonl, seed_everything, write_json
from .model import load_model
from .prompts import messages


def encode_example(tokenizer, row, max_length):
    prompt = messages(row["report"])
    prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(prompt + [{"role": "assistant", "content": row["reference_zh"]}], tokenize=True)
    if full[:len(prefix)] != prefix:
        raise ValueError("Chat template prefix mismatch; refusing an incorrect training mask")
    if len(full) > max_length:
        return None
    labels = [-100] * len(prefix) + full[len(prefix):]
    if all(x == -100 for x in labels):
        raise ValueError("Example has no supervised answer tokens")
    return {"input_ids": full, "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        n = max(len(r["input_ids"]) for r in rows)
        n = (n + 7) // 8 * 8
        ids, labels, masks = [], [], []
        for r in rows:
            pad = n - len(r["input_ids"])
            ids.append(r["input_ids"] + [self.pad_id] * pad)
            labels.append(r["labels"] + [-100] * pad)
            masks.append([1] * len(r["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(masks)}


def train(config_path, data_dir, model_path, output):
    cfg, root, output = load_config(config_path), Path(data_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    output.mkdir(parents=True)
    seed_everything(cfg["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a scheduled CUDA allocation")
    model, tok = load_model(model_path, training=True)
    lora = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
                      lora_dropout=cfg["lora_dropout"], target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    data, filters = {}, {}
    for split in ("train", "validation"):
        raw = read_jsonl(root / f"{split}.zh.jsonl")
        valid = [r for r in raw if r["teacher_accepted"]]
        encoded = [encode_example(tok, r, cfg["max_sequence_length"]) for r in valid]
        data[split] = [r for r in encoded if r is not None]
        filters[split] = {"initial": len(raw), "teacher_rejected": len(raw)-len(valid),
                          "too_long": len(encoded)-len(data[split]), "used": len(data[split])}
    if len(data["train"]) < 512 or len(data["validation"]) < 32:
        raise ValueError("Insufficient accepted data for this pilot")
    collator = Collator(tok.pad_token_id)
    loader = DataLoader(data["train"], batch_size=cfg["train_batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(cfg["seed"]), collate_fn=collator, num_workers=0)
    dev = DataLoader(data["validation"], batch_size=cfg["train_batch_size"], collate_fn=collator, num_workers=0)
    parameters = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], weight_decay=0.01)
    accum = cfg["gradient_accumulation_steps"]
    total_steps = math.ceil(len(loader)/accum) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(opt, int(total_steps * 0.06), total_steps)
    start, history, best, update = time.perf_counter(), [], float("inf"), 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        weighted_loss, tokens, since_log = 0.0, 0, 0.0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            # Last accumulation group can contain fewer than accum microbatches.
            group_size = min(accum, len(loader) - (step // accum)*accum)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            count = int((batch["labels"][:, 1:] != -100).sum())
            weighted_loss += loss.item() * count
            tokens += count
            since_log += loss.item()
            (loss / group_size).backward()
            if (step + 1) % accum == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                update += 1
                if update % 20 == 0:
                    print(f"epoch={epoch} update={update}/{total_steps} token_loss={weighted_loss/max(tokens,1):.5f} elapsed={time.perf_counter()-start:.1f}s", flush=True)
        model.eval()
        dev_loss, dev_tokens = 0.0, 0
        with torch.inference_mode():
            for batch in dev:
                batch = {k: v.to("cuda") for k, v in batch.items()}
                count = int((batch["labels"][:, 1:] != -100).sum())
                dev_loss += model(**batch).loss.item() * count
                dev_tokens += count
        score = dev_loss / max(dev_tokens, 1)
        model.save_pretrained(output / f"epoch-{epoch}")
        tok.save_pretrained(output / f"epoch-{epoch}")
        if score < best:
            best = score
            model.save_pretrained(output / "best")
            tok.save_pretrained(output / "best")
            write_json(output / "selection.json", {"epoch": epoch, "validation_token_loss": score,
                                                    "criterion": "lowest validation answer-token cross-entropy; no final-test selection"})
        history.append({"epoch": epoch, "train_token_loss": weighted_loss/max(tokens, 1),
                        "validation_token_loss": score, "elapsed_seconds": time.perf_counter()-start})
        write_json(output / "history.json", history)
        print(history[-1], flush=True)
    torch.cuda.synchronize()
    write_json(output / "training.json", {"config": cfg, "filters": filters, "history": history,
              "total_parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in parameters), "optimizer_steps": update,
              "elapsed_seconds": time.perf_counter()-start,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pilot.json")
    p.add_argument("--data", default="data/pilot_v1")
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="artifacts/pilot_v1")
    a = p.parse_args()
    train(a.config, a.data, a.model, a.out)
