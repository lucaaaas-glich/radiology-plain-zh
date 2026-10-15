"""Run development/frozen holdout with original or evidence-linked inference.

Synthetic outputs may be committed; this script never reads real patient data.
"""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import mlx.core as mx
from radplain.common import read_jsonl, write_json
from radplain.explain_v2 import explain, render_text
from radplain.mlx_backend import MLXBackend
from radplain.prompts import messages


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--split', choices=['dev','holdout'], default='dev')
    p.add_argument('--mode', choices=['direct','evidence','evidence-no-repair'], default='evidence')
    p.add_argument('--max-tokens', type=int, default=384)
    p.add_argument('--candidate-lock', type=Path)
    a = p.parse_args()
    inputs = Path('examples/challenge.jsonl' if a.split == 'dev' else 'examples/v2/holdout.jsonl')
    if a.split == 'holdout':
        manifest=json.loads(Path('examples/v2/holdout_manifest.json').read_text())
        if digest(inputs) != manifest['sha256']:
            raise ValueError('Frozen holdout changed')
        if not a.candidate_lock:
            p.error('Holdout requires a committed candidate-lock written after development selection')
        lock=json.loads(a.candidate_lock.read_text())
        for file, sha in lock['code_sha256'].items():
            if digest(file) != sha:
                raise ValueError('Frozen candidate changed: '+file)
    a.out.mkdir(parents=True, exist_ok=False)
    source_hashes={str(p):digest(p) for p in sorted(Path('radplain').glob('*.py'))}
    rows = read_jsonl(inputs)
    backend = MLXBackend(a.model)
    results=[]
    with (a.out/'outputs.jsonl').open('x',encoding='utf-8') as f:
        for row in rows:
            if a.mode == 'direct':
                result=backend(messages(row['report']),a.max_tokens)
            else:
                result=explain(row['report'],backend,max_tokens=a.max_tokens,repair=a.mode!='evidence-no-repair')
                result['display_text']=render_text(result)
            r={'id':row['id'],'report':row['report'],'result':result}
            results.append(r)
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
            f.flush()
            print(row['id'],round(result['seconds'],2),flush=True)
    write_json(a.out/'runtime.json',{'split':a.split,'mode':a.mode,'model':a.model,'count':len(rows),
          'input_sha256':digest(inputs),'candidate_lock_sha256':digest(a.candidate_lock) if a.candidate_lock else None,
          'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'load_seconds':backend.load_seconds,'generation_seconds':sum(r['result']['seconds'] for r in results),
          'peak_mlx_allocated_bytes':mx.get_peak_memory(),'platform':platform.platform(),
          'source_files_sha256':source_hashes,
          'source_changed_during_run':any(digest(p)!=sha for p,sha in source_hashes.items())})


if __name__ == '__main__':
    main()
