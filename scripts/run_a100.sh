#!/bin/bash
#SBATCH --job-name=radzh-pilot-v1
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/pilot-%j.out
#SBATCH --error=logs/pilot-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the independent project root}"
PY="$PWD/.venv/bin/python"
TASK_SCRATCH="/scratch/${SLURM_JOB_ID}"
test -d "$TASK_SCRATCH" && test -w "$TASK_SCRATCH"
mkdir -p "$TASK_SCRATCH/radzh-cache" "$TASK_SCRATCH/radzh-models" artifacts results/pilot_v1
export HF_HOME="$TASK_SCRATCH/radzh-cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=180
trap 'code=$?; printf "pipeline_exit=%s job=%s\n" "$code" "$SLURM_JOB_ID" > "logs/pilot-${SLURM_JOB_ID}.exit"' EXIT
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PY" -m pytest -q tests
"$PY" scripts/download_models.py --out "$TASK_SCRATCH/radzh-models"
cp "$TASK_SCRATCH/radzh-models/models.lock.json" results/pilot_v1/models.lock.json
"$PY" scripts/smoke_gpu.py --model "$TASK_SCRATCH/radzh-models/student"
if ! test -f data/pilot_v1/manifest.json; then
  "$PY" -m radplain.data
fi
"$PY" -m radplain.teacher --model "$TASK_SCRATCH/radzh-models/teacher"
# The student is retained in home storage so inference remains possible after scratch cleanup.
if ! test -e artifacts/base-model; then
  mkdir artifacts/base-model
  cp "$TASK_SCRATCH/radzh-models/student/"*.json artifacts/base-model/
  cp "$TASK_SCRATCH/radzh-models/student/"*.safetensors artifacts/base-model/
  cp "$TASK_SCRATCH/radzh-models/student/"*.txt artifacts/base-model/ 2>/dev/null || true
  cp "$TASK_SCRATCH/radzh-models/student/"*.jinja artifacts/base-model/ 2>/dev/null || true
fi
"$PY" -m radplain.train --model artifacts/base-model
# The fixed test is first generated only after dev-loss checkpoint selection is complete.
"$PY" -m radplain.evaluate --model artifacts/base-model --data data/pilot_v1/test.zh.jsonl --out results/pilot_v1/baseline
"$PY" -m radplain.evaluate --model artifacts/base-model --adapter artifacts/pilot_v1/best --data data/pilot_v1/test.zh.jsonl --out results/pilot_v1/lora
"$PY" -m radplain.evaluate --model artifacts/base-model --data examples/challenge.jsonl --out results/pilot_v1/challenge-baseline
"$PY" -m radplain.evaluate --model artifacts/base-model --adapter artifacts/pilot_v1/best --data examples/challenge.jsonl --out results/pilot_v1/challenge-lora
"$PY" scripts/summarize_results.py
date -u
