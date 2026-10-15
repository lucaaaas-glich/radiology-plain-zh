#!/bin/bash
#SBATCH --job-name=radzh-export
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/export-%j.out
#SBATCH --error=logs/export-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2
test -f results/pilot_v1/comparison.json
.venv/bin/python scripts/merge_adapter.py
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib, json
root = Path('artifacts/merged-model')
hashes = {}
for p in sorted(root.glob('*')):
    if not p.is_file():
        continue
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    hashes[p.name] = h.hexdigest()
(root / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
print('Merged export and checksums complete')
PY
