import json
from pathlib import Path
import pytest
from scripts.collect_v3_final import digest
from scripts.resume_v3_remaining import remaining


def fixture(root):
    def write(name,value,lines=False):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in value) if lines else json.dumps(value)+'\n')
    rows=[{'id':'a','report':'alpha'},{'id':'b','report':'beta'}]
    config={'module':'frozen','kwargs':{'use_ledger':True}}
    lock={}
    for name,key,arm in [('10-final20.json','main_request_sha256','v3'),('11-ablation10.json','ablation_request_sha256','without_ledger')]:
        source=f'inputs/{arm}.jsonl';write(source,rows,True)
        req={'input':source,'input_sha256':digest(root/source),'out':f'results/original/{arm}','arms':{arm:config}}
        write('requests/'+name,req);lock[key]=digest(root/'requests'/name)
    write('final_candidate.lock.json',lock)
    def previous(job,arm,subset):
        base=f'results/resume-{job}';source=base+'/input-01.jsonl';write(source,[{k:r[k] for k in ('id','report')} for r in subset],True)
        req={'input':source,'input_sha256':digest(root/source),'out':base+'/part-01','arms':{arm:config}}
        path=base+'/request-01.json';write(path,req)
        write(base+'/plan.json',{'parts':[{'original_request':'10-final20.json' if arm=='v3' else '11-ablation10.json','request':path,'request_sha256':digest(root/path),'ids':[r['id'] for r in subset],'arms':[arm]}]})
        write(req['out']+'/'+arm+'/outputs.jsonl',subset,True)
    return rows,write,previous,digest(root/'final_candidate.lock.json')


def test_reconciles_two_allocations_and_never_retries_completed_failures(tmp_path):
    rows,write,previous,sha=fixture(tmp_path)
    write('results/original/v3/v3/outputs.jsonl',[dict(rows[0],error='original failure')],True)
    previous('1','v3',[dict(rows[1],result={})])
    previous('2','without_ledger',[dict(rows[0],error='continuation failure')])
    tasks,provenance=remaining(tmp_path,sha)
    assert len(tasks)==1 and tasks[0]['rows']==rows[1:]
    assert list(tasks[0]['arms'])==['without_ledger']
    assert provenance['counts']=={'v3':2,'without_ledger':1}


def test_complete_allocation_needs_no_download_or_model(tmp_path):
    rows,write,previous,sha=fixture(tmp_path)
    previous('1','v3',[dict(r,result={}) for r in rows])
    previous('2','without_ledger',[dict(r,error='completed failure') for r in rows])
    tasks,provenance=remaining(tmp_path,sha)
    assert not tasks and provenance['complete']


def test_recovery_rejects_duplicate_or_changed_lock(tmp_path):
    rows,write,previous,sha=fixture(tmp_path)
    with pytest.raises(ValueError,match='lock changed'):remaining(tmp_path,'0'*64)
    previous('1','v3',rows[:1]);previous('2','v3',rows[:1])
    with pytest.raises(ValueError,match='Duplicate completed'):remaining(tmp_path,sha)
