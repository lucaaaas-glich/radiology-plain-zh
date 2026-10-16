"""Recover absent rows after reconciling ALL prior frozen allocations.

Orchestration only: generation code/settings remain fixed. Completed failures
are never retried. The caller must wait for the previous Slurm job to end.
"""
import argparse
import json
import os
from pathlib import Path
from scripts.collect_v3_final import collect, digest
from scripts.resume_v3_final import pending_groups
from scripts.run_v3_worker import execute


def remaining(root, expected_lock_sha):
    lock_path = root/'final_candidate.lock.json'
    if digest(lock_path) != expected_lock_sha:
        raise ValueError('Candidate lock changed')
    completed, provenance = collect(root, lock_path)
    tasks = []
    for name in ('10-final20.json','11-ablation10.json'):
        req = json.loads((root/'requests'/name).read_text())
        rows = [json.loads(s) for s in (root/req['input']).read_text().splitlines() if s.strip()]
        done = {arm: {r['id']:r for r in completed[arm]} for arm in req['arms']}
        for arms, subset in pending_groups(rows, req['arms'], done):
            tasks.append({'original_request':name,'arms':{arm:req['arms'][arm] for arm in arms},'rows':subset})
    return tasks, provenance


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--lock-sha',required=True)
    p.add_argument('--model',type=Path)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args();root=args.root.resolve()
    tasks, provenance=remaining(root,args.lock_sha)
    count=sum(len(t['rows'])*len(t['arms']) for t in tasks)
    print(json.dumps({'completed_rows':provenance['counts'],'remaining_rows':count}),flush=True)
    if args.plan_only:raise SystemExit(0 if tasks else 10)
    if not tasks:return
    if args.model is None:raise ValueError('Model path required')
    job=os.environ['SLURM_JOB_ID'];folder=root/'results'/('resume-'+job)
    folder.mkdir(parents=True,exist_ok=False)
    manifest={'job_id':job,'reconciled_previous_allocations':provenance,
              'remaining_rows':count,'policy':'All prior complete rows retained, including failures; only absent rows; frozen settings','parts':[]}
    for i,task in enumerate(tasks,1):
        source=folder/f'input-{i:02d}.jsonl'
        source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in task['rows']))
        req={'phase':'resume_final_without_selection','input':str(source.relative_to(root)),
             'input_sha256':digest(source),'out':str((folder/f'part-{i:02d}').relative_to(root)),
             'arms':task['arms']}
        path=folder/f'request-{i:02d}.json';path.write_text(json.dumps(req,indent=2)+'\n')
        manifest['parts'].append({'original_request':task['original_request'],'request':str(path.relative_to(root)),
                                 'request_sha256':digest(path),'ids':[r['id'] for r in task['rows']],
                                 'arms':list(task['arms'])})
    (folder/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
    from radplain.hf_backend_v2 import HFBackend
    model=HFBackend(str(args.model),max_input_tokens=8192,load_in_4bit=True)
    for part in manifest['parts']:execute(root/part['request'],model,root)
    (folder/'COMPLETE').write_text('All missing rows executed once; all prior completed failures retained.\n')


if __name__=='__main__':main()
