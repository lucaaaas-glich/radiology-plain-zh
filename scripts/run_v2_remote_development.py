"""A100 development ONLY. No holdout paths are accepted by this script."""
import argparse,hashlib,json,os,platform,subprocess
from importlib.metadata import version
from pathlib import Path
import torch
from radplain.common import read_jsonl,write_json
from radplain.hf_backend_v2 import HFBackend
from radplain.explain_v2 import explain
from radplain.prompts import messages


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    backend=HFBackend(a.model)
    tasks=[('synthetic_direct','examples/challenge.jsonl','direct'),
           ('synthetic_critic','examples/challenge.jsonl','evidence-critic'),
           ('real_critic','data/v2/real/dev.jsonl','evidence-critic')]
    for label,source,mode in tasks:
        rows=[];folder=a.out/label;folder.mkdir()
        with (folder/'outputs.jsonl').open('x',encoding='utf-8') as f:
            for row in read_jsonl(source):
                result=backend(messages(row['report'])) if mode=='direct' else explain(row['report'],backend,max_tokens=384,use_critic=True)
                r={'id':row['id'],'report':row['report'],'result':result}
                rows.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
                print(label,row['id'],round(result['seconds'],2),flush=True)
        write_json(folder/'runtime.json',{'count':len(rows),'mode':mode,'stage':'development_only',
                'generation_seconds':sum(r['result']['seconds'] for r in rows),
                'input_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest()})
    write_json(a.out/'runtime.json',{'model':a.model,'dtype':'bfloat16','device':torch.cuda.get_device_name(),
            'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'load_seconds':backend.load_seconds,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'platform':platform.platform(),
            'packages':{p:version(p) for p in ['torch','transformers','huggingface_hub','safetensors']},
            'source_files_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('radplain').glob('*.py'))}})


if __name__=='__main__':
    main()
