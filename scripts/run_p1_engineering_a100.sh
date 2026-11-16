#!/bin/bash
# One isolated engineering pilot, followed by saved-adapter reload/generation.
# Submit only after checking current user jobs, account and A100 eligibility.
#SBATCH --job-name=radzh-p1-engineering
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=logs/p1-engineering-%j.out
#SBATCH --error=logs/p1-engineering-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Scheduled allocation required}"
: "${SLURM_JOB_ID:?Scheduled allocation required}"
: "${P1_PYTHON:?Explicit isolated P1 environment required}"
: "${P1_BASE_SOURCE:?Read-only existing original base snapshot required}"
: "${P1_RUN_NAME:?New P1 run name required}"
: "${P1_DATA:?Prepared pilot data directory required}"
[[ "$P1_RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || exit 2
P1_TASK_SCRATCH="/scratch/${SLURM_JOB_ID}"
P1_MODEL_STAGE="$P1_TASK_SCRATCH/radzh-p1-base"
test -d "$P1_TASK_SCRATCH"
test ! -e "$P1_MODEL_STAGE"
mkdir "$P1_MODEL_STAGE"
# Read legacy files only. Staging is inside this allocation's private scratch.
for name in config.json generation_config.json tokenizer.json tokenizer_config.json merges.txt vocab.json model.safetensors; do
    cp "$P1_BASE_SOURCE/$name" "$P1_MODEL_STAGE/$name"
done
for name in LICENSE README.md p1_model_provenance.json; do
    cp "artifacts/p1/base-model/$name" "$P1_MODEL_STAGE/$name"
done
export P1_MODEL="$P1_MODEL_STAGE"
export P1_PYTHON P1_DATA P1_RUN_NAME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
"$P1_PYTHON" -m pip freeze > "logs/p1-environment-${SLURM_JOB_ID}.txt"
echo "P1_MODEL_STAGED_FROM_EXISTING_FILES"
bash scripts/run_p1_pilot_a100.sh
# Saved-adapter verification command is fixed before submission below.
"$P1_PYTHON" -m radplain.verify_p1 \
    --run "artifacts/p1/$P1_RUN_NAME" --data "$P1_DATA" --model "$P1_MODEL" \
    --out "results/p1/${P1_RUN_NAME}-verification" \
    --raw-out "data/p1/${P1_RUN_NAME}-verification" \
    --plan examples/p1/pilot_generation_plan_20261117.json
