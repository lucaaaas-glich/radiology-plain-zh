"""Download the pinned original baseline without loading or executing a model."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
FILES = ('config.json', 'generation_config.json', 'tokenizer.json',
         'tokenizer_config.json', 'merges.txt', 'vocab.json', 'LICENSE',
         'README.md', 'model.safetensors')


def checksum(path, spec):
    if path.stat().st_size != spec['size']:
        raise ValueError('Size mismatch: ' + path.name)
    h256, hblob = hashlib.sha256(), hashlib.sha1()
    hblob.update(b'blob ' + str(path.stat().st_size).encode() + b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h256.update(block)
            hblob.update(block)
    if spec.get('lfs_sha256'):
        if h256.hexdigest() != spec['lfs_sha256']:
            raise ValueError('LFS hash mismatch: ' + path.name)
    elif hblob.hexdigest() != spec['git_blob_sha1']:
        raise ValueError('Git blob mismatch: ' + path.name)
    return h256.hexdigest()


def make_lock(path):
    if path.exists():
        raise FileExistsError('Lock already exists')
    url = f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(url, timeout=30) as response:
        metadata = json.load(response)
    if metadata['sha'] != REVISION:
        raise ValueError('Unexpected upstream revision')
    siblings = {x['rfilename']: x for x in metadata['siblings']}
    locked = {'model': MODEL, 'revision': REVISION, 'kind': 'original_base',
              'upstream_metadata_url': url, 'files': {}}
    for name in FILES:
        row = siblings[name]
        locked['files'][name] = {'size': row['size'], 'git_blob_sha1': row['blobId'],
                                 'lfs_sha256': row.get('lfs', {}).get('sha256')}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(locked, stream, indent=2)
        stream.write('\n')


def asset_directory(out):
    """Allow only a direct P1 asset directory, including an unfinished download."""
    root = Path.cwd().resolve()
    bucket = root / 'artifacts/p1'
    out = Path(out)
    if not out.is_absolute():
        out = root / out
    if bucket.resolve() != bucket or out.is_symlink():
        raise ValueError('P1 asset paths must not contain symlinks')
    if out.resolve().parent != bucket:
        raise ValueError('P1 assets must use a direct child directory of artifacts/p1/')
    if out.exists() and not out.is_dir():
        raise ValueError('P1 asset destination must be a directory')
    if out.exists():
        allowed = set(FILES) | {name + '.download-partial' for name in FILES}
        if any(path.is_symlink() for path in out.iterdir()):
            raise ValueError('P1 asset files must not be symlinks')
        if any(path.name not in allowed or not path.is_file() for path in out.iterdir()):
            raise FileExistsError('Refusing a prepared model, adapter, or unrelated existing directory')
    return out


def install_verified_partial(partial, path, spec):
    """Promote an exact upstream match without ever replacing a destination."""
    value = checksum(partial, spec)
    path.hardlink_to(partial)
    partial.unlink()
    return value


def download_file(path, spec):
    partial = path.with_name(path.name + '.download-partial')
    if partial.exists() and partial.stat().st_size == spec['size']:
        return install_verified_partial(partial, path, spec)
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > spec['size']:
                raise ValueError('Partial download is larger than the locked upstream file')
            url = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{path.name}'
            headers = {'User-Agent': 'radplain-p1-asset-prepare'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and (response.status != 206 or
                               not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                    raise ValueError('Server did not honor the partial-download byte range')
                with partial.open('ab' if partial.exists() else 'xb') as stream:
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b''):
                        if stream.tell() + len(block) > spec['size']:
                            raise ValueError('Download exceeds the locked upstream file size')
                        stream.write(block)
            return install_verified_partial(partial, path, spec)
        except Exception as error:
            print(json.dumps({'file': path.name, 'attempt': attempt + 1,
                              'failure_type': type(error).__name__}), flush=True)
            if attempt == 2:
                raise RuntimeError('Pinned asset download failed; details omitted to avoid signed URLs') from None


def prepare(lock_path, out):
    lock = json.loads(lock_path.read_text())
    if (lock['model'], lock['revision'], lock['kind']) != (MODEL, REVISION, 'original_base'):
        raise ValueError('Unexpected original-base lock')
    if set(lock['files']) != set(FILES):
        raise ValueError('Incomplete pinned file set')
    out = asset_directory(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'adapter_config.json').exists() or (out / 'p1_model_provenance.json').exists():
        raise FileExistsError('Model/adapter already prepared; refusing to replace provenance')
    started, hashes = time.perf_counter(), {}
    for name, spec in lock['files'].items():
        path = out / name
        hashes[name] = checksum(path, spec) if path.exists() else download_file(path, spec)
        print(json.dumps({'verified': name, 'bytes': path.stat().st_size}), flush=True)
    provenance = {'model': MODEL, 'revision': REVISION, 'kind': 'original_base',
                  'files_sha256': hashes, 'upstream_verified': True,
                  'upstream_lock_sha256': hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                  'model_loaded': False, 'elapsed_seconds': time.perf_counter() - started}
    with (out / 'p1_model_provenance.json').open('x') as stream:
        json.dump(provenance, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'complete': True, 'files': len(hashes), 'model_loaded': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, default=Path('configs/p1_base_files.lock.json'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/p1/base-model'))
    parser.add_argument('--create-lock', action='store_true')
    args = parser.parse_args()
    if args.create_lock:
        make_lock(args.lock)
    prepare(args.lock, args.out)
