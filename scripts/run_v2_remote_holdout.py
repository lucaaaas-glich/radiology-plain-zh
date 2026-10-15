"""Run a frozen GPU arm only after an explicit local selection request.

The request and final inputs are uploaded after development selection. A GPU
allocation alone never authorizes reading a new holdout or choosing an arm.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from radplain.common import read_jsonl, write_json
from radplain.hf_backend_v2 import HFBackend
from radplain.explain_v2 import explain


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--request', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a=p.parse_args()
    request=json.loads(a.request.read_text())
    lock=json.loads(Path(request['candidate_lock']).read_text())
    arm=lock['arms'][request['arm']]
    if arm['backend']!='cuda-nf4' or arm['mode']!='evidence-critic':
        raise ValueError('Unsupported frozen GPU arm')
    pin=json.loads(Path('results/v2/remote27b.lock.json').read_text())
    if arm['model_repo']!=pin['repo'] or arm['model_revision']!=pin['revision']:
        raise ValueError('Model revision does not match frozen arm')
    for file,sha in lock['code_sha256'].items():
        if digest(file)!=sha:
            raise ValueError('Frozen code changed: '+file)
    for file,sha in request['model_files_sha256'].items():
        if digest(a.model/file)!=sha:
            raise ValueError('Frozen model changed: '+file)
    inputs=[]
    for split in ['synthetic','real']:
        source=Path(request['inputs'][split]['path'])
        if not source.resolve().is_relative_to(Path('final').resolve()):
            raise ValueError('Final inputs must be inside the uploaded final directory')
        if digest(source)!=request['inputs'][split]['sha256']:
            raise ValueError('Frozen data changed: '+split)
        if request['inputs'][split]['sha256']!=lock['holdout_sha256'][split]:
            raise ValueError('Request data differs from selected final holdout')
        inputs.append((split,source))
    a.out.mkdir(parents=True,exist_ok=False)
    backend=HFBackend(a.model,load_in_4bit=True)
    for split,source in inputs:
        folder=a.out/split;folder.mkdir()
        rows=[]
        with (folder/'outputs.jsonl').open('x',encoding='utf-8') as f:
            for row in read_jsonl(source):
                result=explain(row['report'],backend,max_tokens=arm['max_tokens'],use_critic=True)
                value={'id':row['id'],'report':row['report'],'result':result}
                f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();rows.append(value)
                print(split,row['id'],round(result['seconds'],2),flush=True)
        write_json(folder/'runtime.json',{'count':len(rows),'mode':arm['mode'],'arm':request['arm'],
            'generation_seconds':sum(r['result']['seconds'] for r in rows),
            'input_sha256':digest(source),'candidate_lock_sha256':digest(request['candidate_lock']),
            'model_repo':pin['repo'],'model_revision':pin['revision'],'slurm_job_id':os.environ.get('SLURM_JOB_ID')})
    write_json(a.out/'runtime.json',{'load_seconds':backend.load_seconds,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
        'device':torch.cuda.get_device_name(),'quantization':'NF4 double quantization; BF16 compute',
        'candidate_lock_sha256':digest(request['candidate_lock'])})


if __name__=='__main__':
    main()
