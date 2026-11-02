"""Prepare the local candidate lock after development selection, before final outputs.

Does not choose a model, commit files, release the barrier, or run inference.
"""
import argparse
from datetime import datetime, timezone
import json
import inspect
from importlib.metadata import version
from pathlib import Path
import subprocess
from scripts.v2_local_preflight import committed, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--reason-file', required=True, type=Path)
    parser.add_argument('--evidence', required=True, nargs='+', type=Path)
    parser.add_argument('--out', type=Path, default=Path('examples/v2/candidate_local.lock.json'))
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Never overwrite a candidate lock')
    barrier = json.loads(Path('examples/v2/final_inspection_barrier.json').read_text())
    if barrier['state'] != 'held':
        parser.error('Candidate selection must finish before final inspection is released')
    model_records = json.loads(Path('results/v2/models.lock.json').read_text())
    selected = next(r for r in model_records if r['local_path'] == args.model)
    for filename, metadata in selected['files'].items():
        if digest(Path(args.model) / filename) != metadata['sha256']:
            raise ValueError('Downloaded candidate changed: ' + filename)
    evidence = {}
    for path in args.evidence:
        committed(path)
        evidence[str(path)] = digest(path)
    code = [*sorted(Path('radplain').glob('*.py')), Path('scripts/run_v2_pipeline.py'),
            Path('scripts/v2_local_preflight.py'), Path('scripts/freeze_v2_local.py'),
            Path('configs/v2_local.json'), Path('requirements-v2-mac.txt'),
            Path('scripts/prepare_v2_local_model.py')]
    for path in code:
        committed(path)
    config = json.loads(Path('configs/v2_local.json').read_text())
    if config['model'] != args.model or config['mode'] != 'evidence-critic':
        raise ValueError('Local launcher config must match selected evidence-critic arm')
    from radplain.mlx_backend import MLXBackend
    parameters = inspect.signature(MLXBackend).parameters
    for name, expected in {'max_input_tokens': 4096,
                           'memory_limit_bytes': int(5.5*1024**3),
                           'cache_limit_bytes': 128*1024**2}.items():
        if name not in parameters or parameters[name].default != expected:
            raise ValueError('Backend budget differs from candidate manifest: ' + name)
    models = {}
    for model in ('artifacts/mlx-4bit', args.model):
        files = [p for p in Path(model).iterdir() if p.is_file()]
        if not any(p.suffix == '.safetensors' for p in files):
            raise ValueError('Weights missing: ' + model)
        models[model] = {str(p): digest(p) for p in sorted(files)}
    local = {'backend': 'mlx', 'temperature': 0, 'thinking': False,
             'max_input_tokens': 4096, 'mlx_memory_guideline_bytes': int(5.5*1024**3),
             'mlx_cache_limit_bytes': 128*1024**2}
    arms = {
        'v1_direct': dict(local, model='artifacts/mlx-4bit', mode='direct', max_tokens=1024),
        'local_direct': dict(local, model=args.model, mode='direct', max_tokens=1024),
        'local_evidence': dict(local, model=args.model, mode='evidence-critic', max_tokens=384,
                               critic_max_tokens=512, context_policy='all_previous_source_sentences'),
    }
    packages = {}
    for line in Path('requirements-v2-mac.txt').read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            package, expected = line.strip().split('==')
            packages[package] = version(package)
            if packages[package] != expected:
                raise ValueError('Installed package differs from measured requirements: ' + package)
    result = {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
              'stage': 'Local selection based on development only; final predictions not inspected',
              'development_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'selection_reason_zh': args.reason_file.read_text().strip(),
              'selected_model_provenance': selected, 'arms': arms,
              'code_sha256': {str(p): digest(p) for p in code},
              'model_files_sha256': models,
              'installed_packages_at_freeze': packages,
              'holdout_sha256': json.loads(Path('examples/v2/candidate_gpu.lock.json').read_text())['holdout_sha256'],
              'development_evidence': evidence,
              'interpretation': 'Project AI review, not independent or clinical validation; GPU and local evidence pipelines differ.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(str(args.out))


if __name__ == '__main__':
    main()
