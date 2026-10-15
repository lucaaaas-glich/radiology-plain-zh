"""Export a merged student for local MLX conversion; run in a GPU allocation."""
import argparse
from pathlib import Path
from radplain.model import load_model
from radplain.common import write_json

p = argparse.ArgumentParser()
p.add_argument("--base", default="artifacts/base-model")
p.add_argument("--adapter", default="artifacts/pilot_v1/best")
p.add_argument("--out", default="artifacts/merged-model")
a = p.parse_args()
if Path(a.out).exists():
    raise FileExistsError(a.out)
model, tokenizer = load_model(a.base, adapter=a.adapter)
model = model.merge_and_unload(safe_merge=True)
model.save_pretrained(a.out, safe_serialization=True, max_shard_size="2GB")
tokenizer.save_pretrained(a.out)
write_json(Path(a.out)/"export.json", {"base": a.base, "adapter": a.adapter, "operation": "PEFT safe_merge"})
