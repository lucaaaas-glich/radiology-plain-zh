"""Descriptive process/cost audit, never a medical correctness estimator."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
from scripts.summarize_v2 import load_rows


def analyze(rows):
    stages=defaultdict(lambda:{'calls':0,'seconds':0,'input_tokens':0,'generated_tokens':0})
    statuses=Counter();formats=Counter();last_audits=Counter()
    batches=bad_batches=affected_units=without=with_ledger=inapplicable=0
    for row in rows:
        if 'result' not in row:continue
        result=row['result']
        calls=result['calls'];account=row['generation_accounting']['calls']
        if len(calls)!=len(account):raise ValueError('V3 stage calls do not match backend accounting')
        if sum(c['generation_tokens'] for c in calls)!=sum(c['generation_tokens'] for c in account):
            raise ValueError('V3 token totals do not match independent backend accounting')
        for call in calls:
            s=stages[call['stage']];s['calls']+=1;s['seconds']+=call['seconds']
            s['input_tokens']+=call['input_tokens'];s['generated_tokens']+=call['generation_tokens']
        for batch in result['ledger_batches']:
            if batch['call_index'] is not None:
                batches+=1;bad_batches+=bool(batch['errors']);affected_units+=len(batch['errors'])
        seen=set()
        for segment in result['segments']:
            statuses[segment['ledger_status']]+=1
            if segment['status']=='generated':
                if segment['ledger'] is None:without+=1
                else:with_ledger+=1
            for attempt in segment['attempts']:
                key=(attempt['call_start'],attempt['call_end'])
                if key in seen:continue
                seen.add(key)
                if 'error' in attempt:formats[attempt['error']]+=1
                inapplicable+=len(attempt.get('inapplicable_ledger_issues',[]))
            if segment['status']=='source_only' and segment['attempts']:
                last=segment['attempts'][-1]
                for issue in last.get('audit',{}).get('issues',[]):
                    if issue['id']==segment['id']:last_audits[issue['type']]+=1
    return {'reports':len(rows),'stage_cost':dict(stages),'ledger_statuses':dict(statuses),
            'ledger_extraction_batches':batches,'batches_with_format_or_quote_failure':bad_batches,
            'units_affected_by_ledger_format_or_quote_failure':affected_units,
            'generated_with_available_ledger':with_ledger,'generated_from_source_without_ledger':without,
            'write_or_audit_format_errors_by_unique_attempt':dict(formats),
            'inapplicable_complaints_about_missing_ledger':inapplicable,
            'last_audit_issue_types_on_source_only_units':dict(last_audits),
            'note':'These are recorded program/model events, not factual error counts. Available ledger means quoted/accepted, not medically correct. Audit issue types may be wrong or inapplicable; they are not independently validated causes.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--outputs',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Refusing to overwrite analysis')
    result=analyze(load_rows(a.outputs));result['outputs_sha256']=hashlib.sha256(a.outputs.read_bytes()).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
