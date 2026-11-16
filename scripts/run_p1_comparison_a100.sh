#!/bin/bash
#SBATCH --job-name=radzh-p1-comparison
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/p1-comparison-%j.out
#SBATCH --error=logs/p1-comparison-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Scheduled allocation required}"
: "${SLURM_JOB_ID:?Scheduled allocation required}"
: "${P1_PYTHON:?Explicit isolated P1 environment required}"
: "${P1_BASE_SOURCE:?Read-only existing original base required}"
P1_TASK_SCRATCH="/scratch/${SLURM_JOB_ID}"
P1_MODEL_STAGE="$P1_TASK_SCRATCH/radzh-p1-comparison-base"
test -d "$P1_TASK_SCRATCH"
test ! -e "$P1_MODEL_STAGE"
mkdir "$P1_MODEL_STAGE"
for name in config.json generation_config.json tokenizer.json tokenizer_config.json merges.txt vocab.json model.safetensors; do
 cp "$P1_BASE_SOURCE/$name" "$P1_MODEL_STAGE/$name"
done
for name in LICENSE README.md p1_model_provenance.json; do
 cp "artifacts/p1/base-model/$name" "$P1_MODEL_STAGE/$name"
done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"
"$P1_PYTHON" -m pip freeze > "logs/p1-environment-${SLURM_JOB_ID}.txt"
"$P1_PYTHON" scripts/run_p1_comparison.py --model "$P1_MODEL_STAGE"
