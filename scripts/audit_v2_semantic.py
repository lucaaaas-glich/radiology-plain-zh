"""Check a candidate critic on development diagnostic pairs, with saved output."""
import argparse
import json
from pathlib import Path
from radplain.mlx_backend import MLXBackend
from radplain.semantic_v2 import critique
from radplain.common import write_json
from scripts.audit_v2_guard import PAIRS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    model=MLXBackend(a.model)
    rows=[]
    with (a.out/'outputs.jsonl').open('x',encoding='utf-8') as f:
        for key,problem,source,prediction in PAIRS:
            result=critique(source,prediction,model)
            row={'id':key,'expected_problem':problem,'source':source,'prediction':prediction,'result':result}
            rows.append(row)
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            print(key,result['verdict'],flush=True)
    write_json(a.out/'summary.json',{
        'stage':'handwritten development diagnostics, not clinical validation','model':a.model,
        'problem_pairs':sum(r['expected_problem'] for r in rows),
        'problem_pairs_rejected':sum(r['expected_problem'] and not r['result']['verdict']['supported'] for r in rows),
        'valid_pairs':sum(not r['expected_problem'] for r in rows),
        'valid_pairs_rejected':sum(not r['expected_problem'] and not r['result']['verdict']['supported'] for r in rows),
        'invalid_verdicts':sum(not r['result']['verdict']['valid'] for r in rows),
        'seconds':sum(r['result']['generation']['seconds'] for r in rows)})


if __name__=='__main__':
    main()
