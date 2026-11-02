#!/bin/bash
#SBATCH --job-name=radzh-v2-dev27b
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/v2-large-%j.out
#SBATCH --error=logs/v2-large-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
TASK_SCRATCH="/scratch/${SLURM_JOB_ID}"
test -d "$TASK_SCRATCH" && test -w "$TASK_SCRATCH"
export HF_HOME="$TASK_SCRATCH/radzh-v2-large-hf"
export TMPDIR="$TASK_SCRATCH" PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=180
PY="/home/Student/s4905914/radiology-plain-zh/.venv-v2-large/bin/python"
if ! test -x "$PY"; then
  /home/Student/s4905914/miniconda3/envs/torch/bin/python -m venv --system-site-packages /home/Student/s4905914/radiology-plain-zh/.venv-v2-large
fi
trap 'code=$?; printf "pipeline_exit=%s job=%s\n" "$code" "$SLURM_JOB_ID" > "logs/v2-large-${SLURM_JOB_ID}.exit"' EXIT
"$PY" -m pip install --no-cache-dir -r requirements-v2-large.txt
"$PY" -m pytest -q tests/test_numeric_review.py tests/test_v2_guards.py tests/test_v2_quantities.py tests/test_v2_semantic.py tests/test_v2_outcome_counts.py tests/test_v2_precision.py
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
MODEL_DIR="$TASK_SCRATCH/radzh-v2-27b"
export MODEL_DIR
"$PY" - <<'PY'
import os,json,hashlib
from pathlib import Path
from huggingface_hub import snapshot_download
lock=json.loads(Path('results/v2/remote27b.lock.json').read_text())
root=Path(os.environ['MODEL_DIR'])
snapshot_download(lock['repo'],revision=lock['revision'],local_dir=root,max_workers=4,
                  allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','README.md','LICENSE*'])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
files=[p for p in root.iterdir() if p.is_file()]
manifest=dict(lock,files_sha256={p.name:sha(p) for p in files},files_bytes={p.name:p.stat().st_size for p in files})
Path('results/v2/remote27b_download.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('MODEL_DOWNLOAD_VERIFIED',sum(p.stat().st_size for p in files),flush=True)
PY
"$PY" scripts/run_v2_remote_development.py --model "$MODEL_DIR" --out "results/v2/remote27b-${SLURM_JOB_ID}" --load-in-4bit
printf 'DEVELOPMENT_COMPLETE Awaiting a frozen final request for at most20minutes\n'
mkdir -p final
for attempt in $(seq 1 120); do
  if test -f final/skip; then
    printf 'Final comparison skipped by project selection\n'
    exit 0
  fi
  if test -f final/request.json; then
    "$PY" scripts/run_v2_remote_holdout.py --model "$MODEL_DIR" --request final/request.json --out "results/v2/final27b-${SLURM_JOB_ID}"
    exit 0
  fi
  sleep 10
done
printf 'No frozen final request received; released allocation\n'
