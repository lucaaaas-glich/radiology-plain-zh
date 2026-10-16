"""One allocated model, sequential immutable request files, no shell commands."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def inside(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path outside isolated V3 run')
    return path


def load_pipeline(spec, root):
    directory = inside(root/spec['directory'], root)
    for name, sha in spec['code_sha256'].items():
        path = inside(directory/name, directory)
        if digest(path) != sha:
            raise ValueError('Frozen inference code mismatch: '+name)
    label = 'isolated_' + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    package = directory/'radplain'
    module_spec = importlib.util.spec_from_file_location(label, package/'__init__.py', submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[label] = module
    module_spec.loader.exec_module(module)
    return importlib.import_module(label+'.'+spec['module']).explain


def execute(request_path, model, root):
    request = json.loads(request_path.read_text())
    source = inside(root/request['input'], root)
    if digest(source) != request['input_sha256']:
        raise ValueError('Frozen input mismatch')
    out = inside(root/request['out'], root)
    out.mkdir(parents=True, exist_ok=False)
    (out/'request.json').write_bytes(request_path.read_bytes())
    pipelines = {name: load_pipeline(spec, root) for name, spec in request['arms'].items()}
    rows = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    seen = set()
    for r in rows:
        if r['id'] in seen:
            raise ValueError('Duplicate input ID')
        seen.add(r['id'])
    streams = {}
    stats = {}
    try:
        for arm in pipelines:
            folder = out/arm
            folder.mkdir()
            streams[arm] = (folder/'outputs.jsonl').open('x')
            stats[arm] = {'count':0,'seconds':0,'model_calls':0,'input_tokens':0,'generation_tokens':0}
        for row in rows:
            for arm, pipeline in pipelines.items():
                model.max_input_tokens = request['arms'][arm].get('max_input_tokens', 4096)
                calls = []
                def generate(messages, max_tokens):
                    reply = model(messages, max_tokens=max_tokens)
                    calls.append({k:reply[k] for k in ('seconds','input_tokens','generation_tokens','finish_reason')})
                    return reply
                start = time.perf_counter()
                try:
                    result = pipeline(row['report'], generate, **request['arms'][arm].get('kwargs', {}))
                    value = {'id':row['id'],'report':row['report'],'result':result}
                except Exception as error:
                    value = {'id':row['id'],'report':row['report'],'error':type(error).__name__+': '+str(error)}
                value['generation_accounting'] = {'seconds':time.perf_counter()-start,'calls':calls}
                streams[arm].write(json.dumps(value, ensure_ascii=False)+'\n')
                streams[arm].flush()
                stats[arm]['count'] += 1
                stats[arm]['seconds'] += value['generation_accounting']['seconds']
                stats[arm]['model_calls'] += len(calls)
                stats[arm]['input_tokens'] += sum(c['input_tokens'] for c in calls)
                stats[arm]['generation_tokens'] += sum(c['generation_tokens'] for c in calls)
                print('ROW', request_path.name, arm, row['id'], round(value['generation_accounting']['seconds'], 2),
                      value.get('result',{}).get('coverage',{}),value.get('error',''),flush=True)
                (out/'progress.json').write_text(json.dumps(stats, indent=2)+'\n')
        import torch
        runtime = {'arms':stats,'request_sha256':digest(request_path),'input_sha256':digest(source),
                   'job_id':os.environ.get('SLURM_JOB_ID'),'completed_at':datetime.now(timezone.utc).isoformat(),
                   'model_load_seconds':model.load_seconds,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                   'device':torch.cuda.get_device_name(),'worker_sha256':digest(__file__)}
        (out/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    finally:
        for stream in streams.values():
            stream.close()


def main():
    from radplain.hf_backend_v2 import HFBackend
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--deadline',type=float,required=True)
    a=p.parse_args()
    model=HFBackend(a.model,max_input_tokens=8192,load_in_4bit=True)
    print('MODEL_LOADED',round(model.load_seconds,2),flush=True)
    done=set()
    queue=a.root/'requests';queue.mkdir(exist_ok=True)
    while time.time()<a.deadline:
        if (queue/'STOP').exists():
            print('STOP_REQUESTED',flush=True)
            break
        for request in sorted(queue.glob('*.json')):
            if request.name in done:
                continue
            done.add(request.name)
            try:
                execute(request,model,a.root)
                (queue/(request.name+'.done')).write_text('complete\n')
                print('REQUEST_COMPLETE',request.name,flush=True)
            except Exception as error:
                (queue/(request.name+'.failed')).write_text(type(error).__name__+': '+str(error)+'\n')
                print('REQUEST_FAILED',request.name,type(error).__name__,str(error),flush=True)
        time.sleep(5)


if __name__=='__main__':
    main()
