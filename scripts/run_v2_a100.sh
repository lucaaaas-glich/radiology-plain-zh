#!/bin/bash
#SBATCH --job-name=radzh-v2-dev9b
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/v2-dev9b-%j.out
#SBATCH --error=logs/v2-dev9b-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
TASK_SCRATCH="/scratch/${SLURM_JOB_ID}"
test -d "$TASK_SCRATCH" && test -w "$TASK_SCRATCH"
export HF_HOME="$TASK_SCRATCH/radzh-v2-hf"
export TMPDIR="$TASK_SCRATCH"
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=180
PY="/home/Student/s4905914/radiology-plain-zh/.venv-v2/bin/python"
if ! test -x "$PY"; then
  /home/Student/s4905914/miniconda3/envs/torch/bin/python -m venv --system-site-packages /home/Student/s4905914/radiology-plain-zh/.venv-v2
fi
trap 'code=$?; printf "pipeline_exit=%s job=%s\n" "$code" "$SLURM_JOB_ID" > "logs/v2-dev9b-${SLURM_JOB_ID}.exit"' EXIT
"$PY" -m pip install --no-cache-dir -r requirements-v2-gpu.txt
"$PY" -m pytest -q tests/test_numeric_review.py tests/test_v2_guards.py tests/test_v2_quantities.py tests/test_v2_semantic.py
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
MODEL_DIR="$TASK_SCRATCH/radzh-v2-9b"
export MODEL_DIR
"$PY" - <<'PY'
import os,json
from pathlib import Path
from huggingface_hub import snapshot_download
lock=json.loads(Path('results/v2/remote9b.lock.json').read_text())
snapshot_download(lock['repo'],revision=lock['revision'],local_dir=os.environ['MODEL_DIR'],max_workers=4,
                  allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','README.md','LICENSE*'])
PY
"$PY" scripts/run_v2_remote_development.py --model "$MODEL_DIR" --out "results/v2/remote9b-${SLURM_JOB_ID}"
