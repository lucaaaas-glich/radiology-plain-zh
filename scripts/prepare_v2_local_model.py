"""Restore or verify one recorded MLX model without rewriting experiment locks."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def prepare(record, root, verify_only=False, download=None):
    root = root.resolve()
    target = (root / record['local_path']).resolve()
    if not target.is_relative_to(root / 'artifacts' / 'v2'):
        raise ValueError('Model destination must be inside artifacts/v2')
    files = record['files']
    if not any(name.endswith('.safetensors') for name in files):
        raise ValueError('Recorded model has no weights')
    missing = []
    # Check every existing file before downloading anything. Never overwrite a
    # different model, including one currently loaded by another process.
    for name, metadata in files.items():
        if Path(name).name != name:
            raise ValueError('Only recorded root filenames are supported')
        path = target / name
        if not path.exists():
            missing.append(name)
        elif (path.stat().st_size != metadata['bytes']
              or sha256(path) != metadata['sha256']):
            raise ValueError('Existing model file differs; preserve it and use a fresh clone: ' + name)
    if missing and verify_only:
        raise ValueError('Missing model files: ' + ', '.join(missing))
    if missing:
        if download is None:
            from huggingface_hub import hf_hub_download
            download = hf_hub_download
        for name in missing:
            download(repo_id=record['repo'], filename=name,
                     revision=record['revision'], local_dir=target)
            path = target / name
            metadata = files[name]
            if (path.stat().st_size != metadata['bytes']
                    or sha256(path) != metadata['sha256']):
                raise ValueError('Downloaded file failed the frozen hash check: ' + name)
    return {'model': record['local_path'], 'verified_files': len(files),
            'downloaded_files': len(missing), 'revision': record['revision']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='artifacts/v2/qwen35-4b')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    records = json.loads((root / 'results/v2/models.lock.json').read_text())
    matches = [r for r in records if r['local_path'] == args.model]
    if len(matches) != 1:
        parser.error('Choose exactly one recorded local model path')
    print(json.dumps(prepare(matches[0], root, args.verify_only)))


if __name__ == '__main__':
    main()
