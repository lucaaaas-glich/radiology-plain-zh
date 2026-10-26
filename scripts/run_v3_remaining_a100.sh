#!/bin/bash
# Submit ONLY with --dependency=afterany:<previous V3 job> after checking squeue.
#SBATCH --job-name=radzh-v3-tail
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/v3-tail-%j.out
#SBATCH --error=logs/v3-tail-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
test -n "${RADZH_PREVIOUS_JOB_ID:?}"
if test -n "$(/opt/slurm/bin/squeue -h -j "$RADZH_PREVIOUS_JOB_ID")"; then
  echo 'Previous allocation has not ended; refusing overlapping recovery.' >&2
  exit 1
fi
TASK_SCRATCH="/scratch/${SLURM_JOB_ID}"
test -d "$TASK_SCRATCH" && test -w "$TASK_SCRATCH"
export HF_HOME="$TASK_SCRATCH/radzh-v3-hf" TMPDIR="$TASK_SCRATCH" PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=180
PY="/home/Student/s4905914/radiology-plain-zh/.venv-v2-large/bin/python"
set +e
"$PY" -m scripts.resume_v3_remaining --root "$PWD" --lock-sha 21ca039011bbb539a3aaf5fd49c2ea9b8f78e692f8f8665a923237081ab728d0 --plan-only > "logs/resume-plan-${SLURM_JOB_ID}.json"
TASK_PLAN_STATUS=$?
set -e
if test "$TASK_PLAN_STATUS" = 10; then exit 0; fi
test "$TASK_PLAN_STATUS" = 0
MODEL_DIR="$TASK_SCRATCH/radzh-v3-27b"
export MODEL_DIR
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PY" - <<'PY'
import os,json,hashlib
from pathlib import Path
from huggingface_hub import snapshot_download
lock=json.loads(Path('final_candidate.lock.json').read_text())
model=lock['model'];expected=lock['model_files_sha256'][model['model_repo']]
root=Path(os.environ['MODEL_DIR'])
snapshot_download(model['model_repo'],revision=model['model_revision'],local_dir=root,
                  max_workers=4,allow_patterns=list(expected))
for name,digest in expected.items():
    h=hashlib.sha256()
    with (root/name).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=digest:raise ValueError('Continuation model hash mismatch')
print('CONTINUATION_MODEL_HASHES_VERIFIED',flush=True)
PY
"$PY" -m scripts.resume_v3_remaining --root "$PWD" --lock-sha 21ca039011bbb539a3aaf5fd49c2ea9b8f78e692f8f8665a923237081ab728d0 --model "$MODEL_DIR"
