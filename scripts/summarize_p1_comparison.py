"""Aggregate source-bound, candidate-blinded AI reviews without lexical scoring."""
import hashlib,json,math,random,re
from collections import Counter
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [c-h,c+h]
def paired_bootstrap(a,b):
 assert len(a)==len(b);n=len(a);rng=random.Random(3710261117);d=[int(y)-int(x) for x,y in zip(a,b)];samples=sorted(sum(d[rng.randrange(n)] for _ in range(n))/n for _ in range(2000))
 return {'difference':sum(d)/n,'percentile_95_interval':[samples[49],samples[1949]],'resamples':2000,'unit':'paired report','training_seed_replicates':1}
def main():
 root=Path.cwd();local=root/'data/p1/comparison-20261117/blind-review';out=root/'results/p1/comparison-20261117/final_ai_comparison.json'
 if out.exists():raise FileExistsError('Retain first aggregate; corrections require named new artifact')
 receipt=json.loads((root/'results/p1/comparison-20261117/blind_packet_receipt.json').read_text())
 for path,digest in receipt['files_sha256'].items():
  if sha(root/path)!=digest:raise ValueError('Frozen blind packet or mapping changed')
 factlock=json.loads((root/'examples/p1/final_source_facts_20261117.lock.json').read_text())
 for path,digest in factlock['source_review_files_sha256'].items():
  if sha(root/path)!=digest:raise ValueError('Frozen source facts changed')
 mapping=json.loads((local/'arm_mapping.private.json').read_text());assert len(mapping)==len({(r['id'],r['candidate']) for r in mapping})==120
 mp={(r['id'],r['candidate']):r for r in mapping}
 data=[];bindings={}
 for suffix in ['01_14','15_27','28_40']:
  packetpath=local/f'packet_{suffix}.json';reviewpath=local/f'review_{suffix}.json'
  packets=json.loads(packetpath.read_text());reviews=json.loads(reviewpath.read_text());assert len(packets)==len(reviews)
  bindings[str(reviewpath.relative_to(root))]=sha(reviewpath);bindings[str(packetpath.relative_to(root))]=sha(packetpath)
  for p,r in zip(packets,reviews):
   assert hashlib.sha256(p['report'].encode()).hexdigest()==p['source_sha256']
   assert p['id']==r['id'] and p['source_sha256']==r['source_sha256'] and p['position']==r['position']
   assert r['human_review'] is False and r['candidate_arm_names_seen'] is False
   required={x['fact_id'] for x in p['source_facts']['required_facts'] if x['required']}
   candidates={x['candidate']:x for x in p['candidates']};assert {x['candidate'] for x in r['candidates']}==set(candidates)=={'A','B','C'}
   for c in r['candidates']:
    pred=candidates[c['candidate']];assert hashlib.sha256(pred['prediction'].encode()).hexdigest()==pred['prediction_sha256']
    m=mp[(r['id'],c['candidate'])];assert c['prediction_sha256']==pred['prediction_sha256']==m['prediction_sha256']
    fs={x['fact_id']:x['status'] for x in c['fact_assessments']};assert set(fs)==required and len(fs)==len(c['fact_assessments'])
    assert all(v in {'faithful','less_precise','missing','contradicted','unjudgeable'} for v in fs.values())
    for key in ['source_error','incomplete','unjudgeable','precision_issue','chinese_complete','retained_source_text','refusal']:assert type(c[key]) is bool
    assert c['readability'] in {'fluent','needs_revision','hard_to_read'}
    assert 'missing' not in fs.values() or c['incomplete']
    assert 'contradicted' not in fs.values() or c['source_error']
    assert 'unjudgeable' not in fs.values() or c['unjudgeable']
    assert 'less_precise' not in fs.values() or c['precision_issue']
    # No-EOS/failed terminal generation is retained as incomplete independently of style.
    incomplete=c['incomplete'] or not pred['generation_completed']
    status='source_error' if c['source_error'] else 'incomplete' if incomplete else 'unjudgeable' if c['unjudgeable'] else 'pass'
    full=status=='pass' and not c['precision_issue'] and c['chinese_complete'] and not c['retained_source_text']
    data.append({'position':r['position'],'id':r['id'],'arm':m['arm'],'source_sha256':r['source_sha256'],'prediction_sha256':c['prediction_sha256'],'primary_status':status,'complete_faithful_chinese':full,**{k:c[k] for k in ['source_error','unjudgeable','precision_issue','chinese_complete','retained_source_text','refusal','readability']},'incomplete':incomplete,'generation_completed':pred['generation_completed'],'error_code':pred['error_code'],'eos_completed':pred['eos_completed'],'fact_status_counts':dict(Counter(fs.values())),'required_fact_count':len(required),'issue_types':[i['type'] if re.fullmatch(r'[a-z][a-z0-9_]{0,63}',i.get('type','')) else 'other_review_issue' for i in c['issues']],'private_notes_sha256':hashlib.sha256(json.dumps({'issues':c['issues'],'notes_zh':c['notes_zh']},sort_keys=True,ensure_ascii=False).encode()).hexdigest(),'fact_labels_for_consistency':fs})
 assert len(data)==120 and len({(r['id'],r['arm']) for r in data})==120
 seen_text={}
 for r in data:
  key=(r['id'],r['prediction_sha256']);labels={k:r[k] for k in ['source_error','unjudgeable','precision_issue','retained_source_text','refusal','readability','fact_labels_for_consistency']}
  if key in seen_text and seen_text[key]!=labels:raise ValueError('Identical predictions have inconsistent reviews at position '+str(r['position']))
  seen_text[key]=labels
 for r in data:del r['fact_labels_for_consistency']
 arms={}
 for arm in ['base','S0','S1']:
  rows=sorted([r for r in data if r['arm']==arm],key=lambda r:r['position']);assert len(rows)==40
  fs=sum((Counter(r['fact_status_counts']) for r in rows),Counter());total=sum(r['required_fact_count'] for r in rows);full=sum(r['complete_faithful_chinese'] for r in rows);passes=sum(r['primary_status']=='pass' for r in rows)
  arms[arm]={'reports':40,'primary_status':dict(Counter(r['primary_status'] for r in rows)),'complete_faithful_chinese':full,'complete_faithful_wilson95':wilson(full,40),'pass_wilson95':wilson(passes,40),'orthogonal_counts':{k:sum(r[k] for r in rows) for k in ['source_error','incomplete','unjudgeable','precision_issue','chinese_complete','retained_source_text','refusal','generation_completed']},'readability':dict(Counter(r['readability'] for r in rows)),'generation_failure_codes':dict(Counter(r['error_code'] for r in rows if r['error_code'])),'eos_completed':sum(r['eos_completed'] for r in rows),'facts':{'required':total,'statuses':dict(fs),'covered':fs['faithful']+fs['less_precise'],'coverage_micro_fraction':(fs['faithful']+fs['less_precise'])/total,'coverage_macro_fraction':sum((r['fact_status_counts'].get('faithful',0)+r['fact_status_counts'].get('less_precise',0))/r['required_fact_count'] for r in rows)/40,'faithful_only':fs['faithful'],'faithful_micro_fraction':fs['faithful']/total,'faithful_macro_fraction':sum(r['fact_status_counts'].get('faithful',0)/r['required_fact_count'] for r in rows)/40}}
 pairs={}
 for a,b in [('base','S0'),('base','S1'),('S0','S1')]:
  x=sorted([r for r in data if r['arm']==a],key=lambda r:r['position']);y=sorted([r for r in data if r['arm']==b],key=lambda r:r['position']);assert [r['id'] for r in x]==[r['id'] for r in y]
  pairs[a+'_to_'+b]={key:paired_bootstrap([r[key] for r in x],[r[key] for r in y]) for key in ['complete_faithful_chinese','source_error','incomplete']}
  pairs[a+'_to_'+b]['full_faithfulness_gained_positions']=[v['position'] for u,v in zip(x,y) if not u['complete_faithful_chinese'] and v['complete_faithful_chinese']]
  pairs[a+'_to_'+b]['full_faithfulness_lost_positions']=[u['position'] for u,v in zip(x,y) if u['complete_faithful_chinese'] and not v['complete_faithful_chinese']]
 d={'purpose':'exploratory_source_consistency_project_AI_review_not_clinical_accuracy','human_reviews':0,'reviewer_arm_blinding':True,'independent_clinical_gold':False,'bindings_sha256':bindings,'arms':arms,'paired_comparisons':pairs,'anonymous_cases':data,'limits':['40 stratified reports,8 long all BIMCV; not patient population or known pretraining independence','Single training seed; paired report bootstrap does not measure training variation','Rules and AI confirmation only change4 selected training IDs,125 common IDs shift positions; no general data quality success claim','No clinical expert or patient readability evaluation; source ambiguities retained; all failures included']}
 out.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(arms,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
