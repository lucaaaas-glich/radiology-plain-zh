"""Fixed-set generation comparison; lexical scores are not clinical safety."""
import argparse
import re
import time
from pathlib import Path
import torch
from .common import chinese_ratio, load_config, read_jsonl, write_json, write_jsonl
from .model import generate_batch, load_model
from .prompts import messages, PROMPT_VERSION


class ChineseTokenizer:
    def tokenize(self, text):
        import jieba
        return [t for t in jieba.lcut(text.lower()) if re.search(r"[\w\u4e00-\u9fff]", t)]


def score_rows(rows):
    from rouge_score.rouge_scorer import RougeScorer
    from sacrebleu.metrics import CHRF
    scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=ChineseTokenizer())
    eligible = [r for r in rows if r.get("teacher_accepted") and r.get("reference_zh")]
    scores = {key: [] for key in ("rouge1", "rouge2", "rougeL")}
    for row in eligible:
        values = scorer.score(row["reference_zh"], row["prediction"])
        for key in scores:
            scores[key].append(values[key].fmeasure)
    metrics = {key: sum(values)/len(values) if values else None for key, values in scores.items()}
    metrics["chrf"] = CHRF().corpus_score([r["prediction"] for r in eligible],
                                         [[r["reference_zh"] for r in eligible]]).score if eligible else None
    metrics.update({"total_rows": len(rows), "synthetic_reference_scored_rows": len(eligible),
                    "chinese_output_fraction": sum(chinese_ratio(r["prediction"]) >= 0.65 for r in rows)/max(1,len(rows)),
                    "generation_completed_fraction": sum(r["generation"]["completed"] for r in rows)/max(1,len(rows)),
                    "mean_output_characters": sum(len(r["prediction"]) for r in rows)/max(1,len(rows)),
                    "interpretation": "Lexical agreement with a synthetic teacher; not clinical accuracy or evidence of safety."})
    return metrics


def evaluate(config_path, data_path, model_path, adapter, out):
    cfg, out = load_config(config_path), Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {out}")
    out.mkdir(parents=True)
    rows = read_jsonl(data_path)
    model, tok = load_model(model_path, adapter=adapter)
    # Warm-up is excluded from generation throughput.
    generate_batch(model, tok, [messages(rows[0]["report"])], max_new_tokens=16)
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start, generated, generation_seconds = time.perf_counter(), [], 0.0
    batch_size = cfg["generation_batch_size"]
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        text, meta = generate_batch(model, tok, [messages(r["report"]) for r in batch],
                                   max_input_tokens=2048, max_new_tokens=cfg["max_target_tokens"])
        generation_seconds += meta[0]["batch_seconds"]
        for row, prediction, info in zip(batch, text, meta):
            generated.append(dict(row, prediction=prediction, generation=info))
        print(f"evaluate {out}: {len(generated)}/{len(rows)}", flush=True)
    write_jsonl(out / "predictions.jsonl", generated)
    metrics = score_rows(generated)
    metrics.update({"model": cfg["student_model"], "adapter": str(adapter) if adapter else None,
                    "prompt_version": PROMPT_VERSION, "generation_seconds": generation_seconds,
                    "wall_seconds_excluding_loading": time.perf_counter()-start,
                    "output_tokens_per_second": sum(r["generation"]["output_tokens"] for r in generated)/max(generation_seconds,1e-9),
                    "batch_size": batch_size, "do_sample": False,
                    "max_new_tokens": cfg["max_target_tokens"],
                    "device": str(model.device),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated() if model.device.type=="cuda" else None,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved() if model.device.type=="cuda" else None})
    write_json(out / "metrics.json", metrics)
    print(metrics, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pilot.json")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    evaluate(a.config, a.data, a.model, a.adapter, a.out)
