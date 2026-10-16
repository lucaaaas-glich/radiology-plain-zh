#!/bin/bash
#SBATCH --job-name=radzh-v3-ledger
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:10:00
#SBATCH --output=logs/v3-%j.out
#SBATCH --error=logs/v3-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
TASK_SCRATCH="/scratch/${SLURM_JOB_ID}"
test -d "$TASK_SCRATCH" && test -w "$TASK_SCRATCH"
export HF_HOME="$TASK_SCRATCH/radzh-v3-hf" TMPDIR="$TASK_SCRATCH" PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=180
PY="/home/Student/s4905914/radiology-plain-zh/.venv-v2-large/bin/python"
test -x "$PY"
MODEL_DIR="$TASK_SCRATCH/radzh-v3-27b"
export MODEL_DIR
trap 'code=$?; printf "pipeline_exit=%s job=%s\n" "$code" "$SLURM_JOB_ID" > "logs/v3-${SLURM_JOB_ID}.exit"' EXIT
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PY" - <<'PY'
import os,json,hashlib
from pathlib import Path
from huggingface_hub import snapshot_download
lock=json.loads(Path('examples/v2/candidate_gpu.lock.json').read_text())
arm=lock['arms']['gpu27b_evidence']
root=Path(os.environ['MODEL_DIR'])
expected=lock['model_files_sha256'][arm['model_repo']]
snapshot_download(arm['model_repo'],revision=arm['model_revision'],local_dir=root,max_workers=4,allow_patterns=list(expected))
for name,sha in expected.items():
    h=hashlib.sha256()
    with (root/name).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=sha:raise ValueError('Model digest mismatch: '+name)
Path('model_verified.json').write_text(json.dumps({'repo':arm['model_repo'],'revision':arm['model_revision'],'files_sha256':expected},indent=2)+'\n')
print('MODEL_HASHES_VERIFIED',flush=True)
PY
"$PY" scripts/run_v3_worker.py --model "$MODEL_DIR" --deadline 1792175310
