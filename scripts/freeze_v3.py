"""Freeze a reviewed development candidate BEFORE any new cohort generation.

Builds an isolated, hash-checked deployment. Does not submit a Slurm job or queue
requests; the final manifest must be committed before those requests are sent.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from scripts.summarize_v2 import load_rows
from scripts.summarize_v3 import validate, verify_cohort


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--development-request', type=Path, required=True)
    p.add_argument('--development-outputs', type=Path, required=True)
    p.add_argument('--development-input', type=Path, required=True)
    p.add_argument('--development-review', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    a=p.parse_args()
    lock_path=Path('examples/v3/final_candidate.lock.json')
    if lock_path.exists() or a.bundle.exists():
        raise ValueError('Refusing to replace frozen candidate or bundle')
    rows=load_rows(a.development_outputs)
    review=json.loads(a.development_review.read_text())
    validate(rows,review)
    request=json.loads(a.development_request.read_text())
    if sha(a.development_input)!=request['input_sha256']:
        raise ValueError('Development input differs from its queued request')
    expected=load_rows(a.development_input)
    if [(r['id'],r['report']) for r in rows]!=[(r['id'],r['report']) for r in expected]:
        raise ValueError('Every predeclared development case must be completed in order')
    spec=request['arms']['v3']
    for file,digest in spec['code_sha256'].items():
        if sha(file)!=digest:raise ValueError('Candidate changed after development: '+file)
    manifest=json.loads(Path('examples/v3/main20_manifest.json').read_text())
    source=Path('data/v3/real/main20.jsonl')
    if sha(source)!=manifest['input_sha256']:raise ValueError('New cohort changed')
    final=load_rows(source)
    verify_cohort(final,manifest,'final')
    baseline_lock=json.loads(Path('examples/v2/candidate_gpu.lock.json').read_text())
    reference=Path('artifacts/v2/gpu-reference-dev6')
    for file,digest in baseline_lock['code_sha256'].items():
        if sha(reference/file)!=digest:raise ValueError('V2 reference changed: '+file)
    activation=Path('results/v3/development/acceleration_activation.json')
    runtime=json.loads(activation.read_text())
    if not runtime.get('same_model_weights') or not runtime.get('both_final_arms_must_share_runtime'):
        raise ValueError('Missing shared runtime verification')
    if not a.selection.read_text().strip():raise ValueError('Missing selection rationale')
    v3=a.bundle/'candidates/final-v3'
    v2=a.bundle/'candidates/final-v2'
    for file in spec['code_sha256']:
        (v3/file).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,v3/file)
    v2_hashes=dict(baseline_lock['code_sha256'])
    for file in v2_hashes:
        (v2/file).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(reference/file,v2/file)
    for name in ['kernel_probe_v3.py','explain_v2_accelerated.py']:
        file='radplain/'+name
        shutil.copy2(file,v2/file);v2_hashes[file]=sha(file)
    (a.bundle/'inputs').mkdir()
    shutil.copy2(source,a.bundle/'inputs/final20.jsonl')
    subset=a.bundle/'inputs/ablation10.jsonl'
    subset.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in final[::2]))
    v3_spec=dict(spec,directory='candidates/final-v3')
    baseline_spec={'directory':'candidates/final-v2','module':'explain_v2_accelerated',
                   'max_input_tokens':4096,'code_sha256':v2_hashes,
                   'kwargs':{'max_tokens':384,'use_critic':True}}
    main_request={'phase':'final','input':'inputs/final20.jsonl','input_sha256':sha(source),
                  'out':'results/final20','arms':{'v2':baseline_spec,'v3':v3_spec}}
    control_spec=dict(v3_spec,kwargs=dict(v3_spec['kwargs'],use_ledger=False))
    control_request={'phase':'ablation','input':'inputs/ablation10.jsonl','input_sha256':sha(subset),
                     'out':'results/ablation10','arms':{'without_ledger':control_spec}}
    for name,value in [('10-final20.json',main_request),('11-ablation10.json',control_request)]:
        write(a.bundle/'requests'/(name+'.pending'),value)
        write(Path('examples/v3')/name,value)
    for name in ('resume_v3_final.py','run_v3_continuation_a100.sh'):
        target=a.bundle/'scripts'/name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(Path('scripts')/name,target)
    dependencies=[a.development_request,a.development_input,a.development_outputs,a.development_review,a.selection,activation,
                  Path('docs/v3/预注册评估方案.md'),Path('examples/v3/fla_package_sources.json'),
                  Path('examples/v3/real_manifest.json'),Path('examples/v3/main20_manifest.json'),
                  Path('docs/v3/测试规模修订.md'),Path('examples/v2/candidate_gpu.lock.json'),
                  Path('scripts/resume_v3_final.py'),Path('scripts/run_v3_continuation_a100.sh')]
    lock={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
          'stage':'Candidate frozen before new cohort predictions; selection used OLD development only',
          'model':baseline_lock['arms']['gpu27b_evidence'],
          'model_files_sha256':baseline_lock['model_files_sha256'],
          'shared_acceleration':runtime,'arms':main_request['arms'],
          'ablation':control_request,'main_request_sha256':sha(Path('examples/v3/10-final20.json')),
          'ablation_request_sha256':sha(Path('examples/v3/11-ablation10.json')),
          'input_sha256':sha(source),'dependencies_sha256':{str(f):sha(f) for f in dependencies},
          'worker_sha256':sha('scripts/run_v3_worker.py'),
          'interpretation':'Same model and accelerated runtime. V2 original algorithm preserved; V3 has structure, glossary, batching, ledger and audit changes. The 10-case ablation only explores the ledger contribution.'}
    write(lock_path,lock)
    shutil.copy2(lock_path,a.bundle/'final_candidate.lock.json')
    with tarfile.open(str(a.bundle)+'.tar.gz','w:gz') as archive:
        for child in a.bundle.iterdir():archive.add(child,arcname=child.name)
    print('Frozen',lock_path,'bundle',str(a.bundle)+'.tar.gz')
    print('Commit the manifest and request files BEFORE queueing the .pending requests.')


if __name__=='__main__':main()
