"""Report-level paired evaluation; exact trace binding and explicit incompleteness."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from scripts.summarize_v2 import fingerprint, CATEGORIES, READABILITY, load_rows
from radplain.common import fingerprint as source_fingerprint


def verify_cohort(rows,manifest,stage):
    expected=manifest['rows'][::2] if stage=='ablation' else manifest['rows']
    if {r['id'] for r in rows}!={r['id'] for r in expected} or len(rows)!=len(expected):
        raise ValueError('Final results do not cover the complete predeclared cohort')
    indexed={r['id']:r for r in expected}
    if any(source_fingerprint(r['report'])!=indexed[r['id']]['source_sha256'] for r in rows):
        raise ValueError('Final source report changed')


def validate(rows, review):
    cases = review['cases']
    if len(cases) != len(rows) or len({r['id'] for r in cases}) != len(cases):
        raise ValueError('One completed review per output is required')
    indexed = {r['id']:r for r in cases}
    if set(indexed) != {r['id'] for r in rows}:
        raise ValueError('Review IDs do not match outputs')
    for row in rows:
        c = indexed[row['id']]
        if c['output_sha256'] != fingerprint(row):
            raise ValueError('Output changed after review: '+row['id'])
        if not isinstance(c['errors'],list) or len(c['errors']) != len(set(c['errors'])) or any(e not in CATEGORIES for e in c['errors']):
            raise ValueError('Invalid error categories')
        for key in ('clinical_coverage_complete','precision_concern'):
            if type(c[key]) is not bool:
                raise ValueError('Missing explicit judgment: '+key)
        if c['readability'] not in READABILITY or not isinstance(c['notes_zh'],str) or not c['notes_zh'].strip():
            raise ValueError('Missing review explanation')
        if 'error' in row and (c['clinical_coverage_complete'] or c['readability']!='none'):
            raise ValueError('Failed generation cannot count as a complete explanation')
        if c['ledger_review'] not in {'not_applicable','not_reviewed','no_observed_issue','observed_issue','not_available'}:
            raise ValueError('Invalid ledger review status')
    return indexed


def faithful(c):
    return c['clinical_coverage_complete'] and not c['errors'] and not c['precision_concern']


def summarize(rows,review):
    indexed=validate(rows,review)
    durations=[r['generation_accounting']['seconds'] for r in rows]
    counts=Counter()
    for c in indexed.values():
        outcome=('observed_error' if c['errors'] else 'complete_faithful' if faithful(c)
                 else 'incomplete_without_observed_error' if not c['clinical_coverage_complete']
                 else 'precision_concern_without_clear_error')
        counts[outcome]+=1
    segments=[s for r in rows for s in r.get('result',{}).get('segments',[])]
    eligible=[s for s in segments if s['status'] not in {'structure','excluded_instruction'}]
    calls=[c for r in rows for c in r['generation_accounting']['calls']]
    return {'reports':len(rows),'not_clinical_accuracy':True,
            'review_scope':review['reviewer'],'mutually_exclusive_outcomes':dict(counts),
            'reports_with_observed_factual_errors':sum(bool(c['errors']) for c in indexed.values()),
            'reports_complete_faithful_chinese':sum(faithful(c) for c in indexed.values()),
            'reports_clinical_coverage_complete':sum(c['clinical_coverage_complete'] for c in indexed.values()),
            'reports_precision_concern':sum(c['precision_concern'] for c in indexed.values()),
            'reports_fully_generated':sum(r.get('result',{}).get('coverage',{}).get('fully_generated',False) for r in rows),
            'generation_failures':sum('error' in r for r in rows),
            'error_category_report_counts':dict(Counter(e for c in indexed.values() for e in c['errors'])),
            'readability_ai_judgments':dict(Counter(c['readability'] for c in indexed.values())),
            'ledger_review_counts':dict(Counter(c['ledger_review'] for c in indexed.values())),
            'ledger_status_counts':dict(Counter(s.get('ledger_status','not_applicable') for s in eligible)),
            'generated_with_ledger':sum(s['status']=='generated' and s.get('ledger') is not None for s in eligible),
            'generated_without_ledger':sum(s['status']=='generated' and s.get('ledger') is None for s in eligible),
            'segments':{'eligible':len(eligible),'generated':sum(s['status']=='generated' for s in eligible),
                        'source_only':sum(s['status']=='source_only' for s in eligible),
                        'structure':sum(s['status']=='structure' for s in segments),
                        'excluded_instruction':sum(s['status']=='excluded_instruction' for s in segments)},
            'latency_seconds':{'total':sum(durations),'median':statistics.median(durations),'minimum':min(durations),'maximum':max(durations)},
            'generation_cost':{'model_calls':len(calls),'input_tokens':sum(c['input_tokens'] for c in calls),
                               'generated_tokens':sum(c['generation_tokens'] for c in calls)},
            'note':'Clinical completeness is an AI review judgment. Fully-generated and per-segment coverage are mechanical; segmentation differs between arms.'}


def compare(rows_a,review_a,rows_b,review_b):
    a,b=validate(rows_a,review_a),validate(rows_b,review_b)
    if set(a)!=set(b):
        raise ValueError('Different paired reports')
    raw_a,raw_b={r['id']:r for r in rows_a},{r['id']:r for r in rows_b}
    if any(raw_a[k]['report']!=raw_b[k]['report'] for k in a):
        raise ValueError('Paired reports have different source text')
    counts={m:Counter() for m in ('observed_error','complete_faithful','clinical_coverage_complete')}
    cases=[]
    for key in a:
        values={'observed_error':(bool(a[key]['errors']),bool(b[key]['errors'])),
                'complete_faithful':(faithful(a[key]),faithful(b[key])),
                'clinical_coverage_complete':(a[key]['clinical_coverage_complete'],b[key]['clinical_coverage_complete'])}
        case={'id':key}
        for metric,(x,y) in values.items():
            state=('both' if y else 'only_a') if x else ('only_b' if y else 'neither')
            counts[metric][state]+=1;case[metric]=state
        cases.append(case)
    return {'paired_reports':len(a),'counts':{k:dict(v) for k,v in counts.items()},'cases':cases,'not_clinical_accuracy':True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--outputs',required=True,type=Path)
    p.add_argument('--review',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--stage',required=True,choices=['development','final','ablation'])
    a=p.parse_args()
    if a.out.exists():raise ValueError('Refusing to replace a summary')
    rows=load_rows(a.outputs)
    if a.stage!='development':
        manifest_path=Path('examples/v3/main20_manifest.json')
        verify_cohort(rows,json.loads(manifest_path.read_text()),a.stage)
        if not Path('examples/v3/final_candidate.lock.json').exists():
            raise ValueError('Final candidate has not been frozen')
    result=summarize(rows,json.loads(a.review.read_text()))
    result['stage']=a.stage
    if a.stage!='development':
        result['cohort_manifest_sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        result['candidate_lock_sha256']=hashlib.sha256(Path('examples/v3/final_candidate.lock.json').read_bytes()).hexdigest()
    result['outputs_file_sha256']=hashlib.sha256(a.outputs.read_bytes()).hexdigest()
    result['review_file_sha256']=hashlib.sha256(a.review.read_bytes()).hexdigest()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
