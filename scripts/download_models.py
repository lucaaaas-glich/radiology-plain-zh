"""Pin model revisions and cache only necessary inference files on job scratch."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/pilot.json")
p.add_argument("--out", required=True)
a = p.parse_args()
cfg = json.loads(Path(a.config).read_text())
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
lock = {}
for kind in ("teacher", "student"):
    name = cfg[kind + "_model"]
    revision = HfApi().model_info(name).sha
    folder = out / kind
    snapshot_download(name, revision=revision, local_dir=folder,
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    lock[kind] = {"model": name, "revision": revision, "path": str(folder)}
    print(kind, name, revision, flush=True)
(out / "models.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
