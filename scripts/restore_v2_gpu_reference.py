"""Reconstruct the measured GPU source without changing this checkout.

Only source files are restored. No weights, report data, network requests, GPU
allocation, or inference are performed. Existing destinations are never replaced.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


def digest(data):
    return hashlib.sha256(data).hexdigest()


def restore(root, out):
    out = out.resolve()
    if out.exists():
        raise ValueError('Refusing to replace an existing directory')
    lock_path = root/'examples/v2/candidate_gpu.lock.json'
    lock = json.loads(lock_path.read_text())
    recipe = lock['reconstruction']
    patch = (root/recipe['patch']).read_bytes()
    if digest(patch) != recipe['patch_sha256']:
        raise ValueError('GPU reconstruction patch changed')
    names = set(lock['code_sha256'])
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe source path')
    for line in patch.decode().splitlines():
        if line.startswith(('--- ', '+++ ')):
            name = line[4:].split('\t')[0]
            if name[:2] not in {'a/', 'b/'} or name[2:] not in names:
                raise ValueError('Patch targets a file outside the frozen source manifest')
    with tempfile.TemporaryDirectory(prefix='radzh-gpu-source-') as temporary:
        temp = Path(temporary)
        for name in sorted(names):
            content = subprocess.check_output(['git', 'show', recipe['base_commit']+':'+name], cwd=root)
            target = temp/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        subprocess.run(['git', 'apply', '--check', '-'], input=patch, cwd=temp, check=True)
        subprocess.run(['git', 'apply', '-'], input=patch, cwd=temp, check=True)
        for name, sha in lock['code_sha256'].items():
            if digest((temp/name).read_bytes()) != sha:
                raise ValueError('Restored GPU source digest mismatch: '+name)
        (temp/'candidate_gpu.lock.json').write_bytes(lock_path.read_bytes())
        (temp/'RESTORED.json').write_text(json.dumps({
            'candidate_lock_sha256': digest(lock_path.read_bytes()),
            'source_files_verified': len(names),
            'note': 'Measured dev6 GPU pipeline. No weights, data, inference or GPU job included.'
        }, indent=2)+'\n')
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(temp, out)
    return {'destination': str(out), 'verified_files': len(names)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('artifacts/v2/gpu-reference-dev6'))
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(restore(root, a.out)))


if __name__ == '__main__':
    main()
