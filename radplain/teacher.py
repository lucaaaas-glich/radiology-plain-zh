"""Create clearly labelled synthetic Chinese targets, resumably by report hash."""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from .common import chinese_ratio, load_config, read_jsonl, write_json, write_jsonl
from .model import generate_batch, load_model
from .prompts import teacher_messages


def translate(config_path, data_dir, model_path):
    cfg, root = load_config(config_path), Path(data_dir)
    model, tokenizer = load_model(model_path)
    stats, start = {}, time.perf_counter()
    for split in ("train", "validation", "test"):
        rows = read_jsonl(root / f"{split}.jsonl")
        journal = root / f"{split}.teacher-journal.jsonl"
        done = {r["id"]: r for r in read_jsonl(journal)} if journal.exists() else {}
        pending = [r for r in rows if r["id"] not in done]
        # Length ordering reduces padding; evaluation membership is unchanged.
        pending.sort(key=lambda r: len(r["report"]) + len(r["reference_en"]))
        batch_size = cfg["teacher_batch_size"]
        with journal.open("a", encoding="utf-8") as f:
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset:offset + batch_size]
                outputs, meta = generate_batch(model, tokenizer, [teacher_messages(r) for r in batch],
                                               max_input_tokens=3072, max_new_tokens=cfg["max_target_tokens"])
                for row, text, info in zip(batch, outputs, meta):
                    accepted = chinese_ratio(text) >= 0.65 and len(text) >= 12 and info["completed"]
                    item = dict(row, reference_zh=text, reference_provenance="synthetic_Qwen_teacher_not_clinician_reviewed",
                                teacher_model=cfg["teacher_model"], teacher_generation=info,
                                teacher_accepted=accepted)
                    done[row["id"]] = item
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                if offset % (batch_size * 16) == 0:
                    print(f"teacher {split}: {len(done)}/{len(rows)}, elapsed={time.perf_counter()-start:.1f}s", flush=True)
        # Keep held-out members even if teacher output fails; mark reference validity.
        ordered = [done[r["id"]] for r in rows]
        write_jsonl(root / f"{split}.zh.jsonl", ordered)
        stats[split] = {"total": len(ordered), "accepted": sum(r["teacher_accepted"] for r in ordered)}
    write_json(root / "teacher_summary.json", {"splits": stats, "elapsed_seconds": time.perf_counter()-start,
                                               "teacher": cfg["teacher_model"],
                                               "warning": "Synthetically generated translations; acceptance checks only language, length and completion, not clinical fidelity."})
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pilot.json")
    p.add_argument("--data", default="data/pilot_v1")
    p.add_argument("--model", required=True)
    a = p.parse_args()
    translate(a.config, a.data, a.model)
