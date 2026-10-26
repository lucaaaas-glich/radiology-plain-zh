"""Resume only missing final rows after the previous Slurm job has ENDED.

Completed error rows are preserved, never retried. Old files stay untouched.
Requests and sources must still match the previously committed final lock.
This is recovery from an interrupted allocation, not a second attempt at cases.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from scripts.run_v3_worker import digest, inside, execute


def completed_rows(path,expected):
    """Ignore only an unfinished final JSON line; reject all other corruption."""
    if not path.exists():return {}
    data=path.read_bytes();lines=data.splitlines(keepends=True);rows={}
    for index,line in enumerate(lines):
        try:row=json.loads(line)
        except (ValueError,UnicodeDecodeError):
            if index==len(lines)-1 and not line.endswith(b'\n'):break
            raise ValueError('Corrupt completed trace: '+str(path))
        key=row['id']
        if key in rows or key not in expected or row['report']!=expected[key]['report']:
            raise ValueError('Duplicate, unknown, or changed output source')
        rows[key]=row
    return rows


def pending_groups(rows,arms,completed):
    """Consecutive requests preserve the original report/arm order."""
    groups=[]
    for row in rows:
        missing=tuple(a for a in arms if row['id'] not in completed[a])
        if not missing:continue
        if groups and groups[-1][0]==missing:groups[-1][1].append(row)
        else:groups.append((missing,[row]))
    return groups


def plan(root):
    lock=json.loads((root/'final_candidate.lock.json').read_text())
    specs=[('10-final20.json',lock['main_request_sha256']),
           ('11-ablation10.json',lock['ablation_request_sha256'])]
    tasks=[];inventory={}
    for name,sha in specs:
        path=root/'requests'/name
        if digest(path)!=sha:raise ValueError('Final request changed')
        request=json.loads(path.read_text());source=inside(root/request['input'],root)
        if digest(source)!=request['input_sha256']:raise ValueError('Final input changed')
        rows=[json.loads(s) for s in source.read_text().splitlines() if s.strip()]
        indexed={r['id']:r for r in rows}
        if len(indexed)!=len(rows):raise ValueError('Duplicate source IDs')
        completed={a:completed_rows(root/request['out']/a/'outputs.jsonl',indexed) for a in request['arms']}
        # A continuation only runs once. A second recovery must explicitly
        # account for its traces rather than silently repeating those outputs.
        if list((root/'results').glob('resume-*/plan.json')):
            raise ValueError('A prior continuation exists; reconcile its traces first')
        inventory[name]={a:len(v) for a,v in completed.items()}
        for arms,subset in pending_groups(rows,request['arms'],completed):
            tasks.append({'original_request':name,'arms':{a:request['arms'][a] for a in arms},'rows':subset})
    return tasks,inventory


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model',type=Path)
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    tasks,inventory=plan(root)
    total=sum(len(t['rows'])*len(t['arms']) for t in tasks)
    print(json.dumps({'completed_original_rows':inventory,'remaining_rows':total}),flush=True)
    if a.plan_only:raise SystemExit(0 if tasks else 10)
    if not tasks:return
    if a.model is None:raise ValueError('Model path required')
    job=os.environ['SLURM_JOB_ID'];folder=root/'results'/('resume-'+job)
    folder.mkdir(parents=True,exist_ok=False)
    manifest={'job_id':job,'completed_original_rows':inventory,'remaining_rows':total,
              'policy':'Only absent rows; completed failures are never retried; originals untouched','parts':[]}
    for i,task in enumerate(tasks,1):
        input_path=folder/f'input-{i:02d}.jsonl'
        input_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in task['rows']))
        req={'phase':'resume_final_without_selection','input':str(input_path.relative_to(root)),
             'input_sha256':digest(input_path),'out':str((folder/f'part-{i:02d}').relative_to(root)),
             'arms':task['arms']}
        path=folder/f'request-{i:02d}.json';path.write_text(json.dumps(req,indent=2)+'\n')
        manifest['parts'].append({'original_request':task['original_request'],'request':str(path.relative_to(root)),
                                  'request_sha256':digest(path),'ids':[r['id'] for r in task['rows']],'arms':list(task['arms'])})
    (folder/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
    from radplain.hf_backend_v2 import HFBackend
    model=HFBackend(str(a.model),max_input_tokens=8192,load_in_4bit=True)
    for part in manifest['parts']:
        execute(root/part['request'],model,root)
    (folder/'COMPLETE').write_text('All missing rows executed once. Original failures retained.\n')


if __name__=='__main__':main()
