"""Two OLD reports only: measure 8GB-Mac feasibility, not a new benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from importlib.metadata import version
from radplain.common import read_jsonl, write_json
from radplain.explain_v3 import explain
from radplain.mlx_backend import MLXBackend


def main():
    import mlx.core as mx
    p=argparse.ArgumentParser()
    p.add_argument('--label',choices=['dev1','dev2checked'],default='dev1')
    a=p.parse_args()
    folder=Path('results/v3/real-data')/('local-'+a.label)
    folder.mkdir(parents=True,exist_ok=False)
    old=read_jsonl('data/v2/real/holdout.jsonl')
    rows=[old[12],old[13]]
    model=Path('artifacts/v2/qwen35-4b')
    hashes={}
    for file in sorted(model.iterdir()):
        if file.is_file() and file.suffix in {'.json','.safetensors','.jinja'}:
            h=hashlib.sha256()
            with file.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            hashes[file.name]=h.hexdigest()
    code={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('radplain').glob('*.py')}
    backend=MLXBackend(str(model),max_input_tokens=8192)
    output=[]
    with (folder/'outputs.jsonl').open('x') as f:
        for row in rows:
            calls=[]
            def generate(messages,max_tokens):
                r=backend(messages,max_tokens=max_tokens)
                calls.append({k:r[k] for k in ('seconds','input_tokens','generation_tokens','finish_reason')})
                return r
            start=time.perf_counter()
            try:
                result=explain(row['report'],generate)
                value={'id':row['id'],'report':row['report'],'result':result}
            except Exception as error:
                value={'id':row['id'],'report':row['report'],'error':str(error)}
            value['generation_accounting']={'seconds':time.perf_counter()-start,'calls':calls}
            f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();output.append(value)
            print(row['id'],value['generation_accounting']['seconds'],value.get('result',{}).get('coverage'),value.get('error',''),flush=True)
    runtime_path='results/v3/local_probe_runtime.json' if a.label=='dev1' else 'results/v3/local_'+a.label+'_runtime.json'
    write_json(runtime_path,{'stage':'development_feasibility_only','label':a.label,
        'model':'mlx-community/Qwen3.5-4B-4bit','model_revision':'0e7ffd5c629ef7719d4cbc04069232580bfa9d9c',
        'count':len(output),'old_v2_case_indices':[13,14],'model_files_sha256':hashes,'code_sha256':code,
        'load_seconds':backend.load_seconds,'peak_mlx_memory_bytes':mx.get_peak_memory(),
        'total_seconds':sum(r['generation_accounting']['seconds'] for r in output),
        'platform':platform.platform(),'packages':{p:version(p) for p in ['mlx','mlx-lm','transformers']},
        'note':'Two already-seen old reports; no claim of unseen-test quality or equivalence to27B.'})


if __name__=='__main__':main()
