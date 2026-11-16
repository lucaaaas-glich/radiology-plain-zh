#!/bin/bash
# Template only. This file never downloads a model, connects by SSH, or submits a job.
# Before manually submitting: inspect existing jobs/allocations, confirm the current
# account/partition/A100 policy, and check that this run will not duplicate a job.
# Supply current --account and --partition explicitly to sbatch; none are assumed here.
# Example AFTER those checks (replace placeholders yourself):
# sbatch --account=<current-account> --partition=<current-partition> \
#   --export=ALL,P1_DATA=<pilot-dir>,P1_MODEL=<original-snapshot>,P1_RUN_NAME=<new-run>,P1_PYTHON=<independent-p1-venv>/bin/python \
#   scripts/run_p1_pilot_a100.sh
#SBATCH --job-name=radzh-p1-pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=p1-pilot-%j.out
#SBATCH --error=p1-pilot-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?This template must run inside an explicitly scheduled allocation}"
: "${SLURM_JOB_ID:?A scheduled Slurm job is required}"
: "${SLURM_JOB_ACCOUNT:?Pass the current account explicitly to sbatch}"
: "${SLURM_JOB_PARTITION:?Pass the current partition explicitly to sbatch}"
: "${P1_DATA:?Set P1_DATA to the prepared pilot train/validation directory}"
: "${P1_MODEL:?Set P1_MODEL to the complete locked original Qwen snapshot}"
: "${P1_RUN_NAME:?Set a fresh P1_RUN_NAME; existing outputs are rejected}"
: "${P1_PYTHON:?Set P1_PYTHON to an explicitly prepared independent P1 GPU environment}"
[[ "$P1_RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "Invalid P1_RUN_NAME" >&2; exit 2; }
PY="$P1_PYTHON"
CONFIG="${P1_CONFIG:-configs/p1_pilot.json}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD"
test -x "$PY"
test -d "$P1_DATA"
test -d "$P1_MODEL"
test ! -e "results/p1/${P1_RUN_NAME}-preflight"
test ! -e "artifacts/p1/${P1_RUN_NAME}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PY" -m radplain.train_p1 preflight --mode pilot --config "$CONFIG" \
    --data "$P1_DATA" --tokenizer "$P1_MODEL" --out "results/p1/${P1_RUN_NAME}-preflight"
"$PY" -m radplain.train_p1 train --mode pilot --config "$CONFIG" \
    --data "$P1_DATA" --tokenizer "$P1_MODEL" --model "$P1_MODEL" --out "artifacts/p1/$P1_RUN_NAME"
# Deliberately ends after dev-only selection. Pilot results are engineering checks;
# this script does not run final-test inference or claim formal effectiveness.
