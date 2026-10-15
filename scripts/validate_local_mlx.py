"""Record actual quantized Mac generation separately from A100 results."""
import argparse
import json
import platform
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from radplain.common import read_jsonl, write_json, write_jsonl
from radplain.prompts import messages

p = argparse.ArgumentParser()
p.add_argument("--model", default="artifacts/mlx-4bit")
p.add_argument("--out", default="results/mac_4bit_v1")
a = p.parse_args()
out = Path(a.out)
if out.exists():
    raise FileExistsError(out)
rows = read_jsonl("examples/challenge.jsonl")
model, tok = load(a.model)
start, results = time.perf_counter(), []
for row in rows:
    prompt = tok.apply_chat_template(messages(row["report"]), tokenize=False, add_generation_prompt=True)
    before, parts, last = time.perf_counter(), [], None
    for r in stream_generate(model, tok, prompt=prompt, max_tokens=384, sampler=make_sampler(temp=0)):
        parts.append(r.text)
        last = r
    results.append(dict(row, prediction="".join(parts).strip(), seconds=time.perf_counter()-before,
                        generation_tokens=getattr(last, "generation_tokens", None),
                        finish_reason=getattr(last, "finish_reason", None)))
    print(row["id"], results[-1]["seconds"], flush=True)
write_jsonl(out/"predictions.jsonl", results)
write_json(out/"runtime.json", {"platform": platform.platform(), "processor": platform.machine(),
           "quantization": "MLX affine4bit group64", "elapsed_seconds": time.perf_counter()-start,
           "peak_memory_bytes": mx.get_peak_memory(), "count": len(rows),
           "note": "Local quantized challenge inference, including first-run compilation; not the A100 benchmark."})
