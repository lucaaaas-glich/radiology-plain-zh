"""Declared posthoc annotation-scope sensitivities; primary labels stay intact."""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path

def main():
 root=Path.cwd();base=root/'data/p1/comparison-20261117/blind-review';public=root/'results/p1/comparison-20261117'
 dest=public/'annotation_sensitivity.json'
 if dest.exists():raise FileExistsError('Retain first sensitivity artifact')
 plan=json.loads((public/'source_annotation_sensitivity_plan_extended.json').read_text());primary=json.loads((public/'final_ai_comparison.json').read_text())
 mapping={(r['id'],r['candidate']):r['arm'] for r in json.loads((base/'arm_mapping.private.json').read_text())}
 records=[]
 for suffix in ['01_14','15_27','28_40']:
  for r in json.loads((base/f'review_{suffix}.json').read_text()):
   for c in r['candidates']:records.append((r,c,mapping[(r['id'],c['candidate'])]))
 result={}
 for name,extra in [('narrow',[]),('broad',plan['broad_additional_remove'])]:
  excluded={(r['id'],r['fact_id']) for r in plan['narrow_remove']+extra};arms={}
  for arm in ['base','S0','S1']:
   counts=Counter();macros=[];faithfulmacros=[];affected=[]
   for r,c,a in records:
    if a!=arm:continue
    facts=[f for f in c['fact_assessments'] if (r['id'],f['fact_id']) not in excluded];n=len(facts);current=Counter(f['status'] for f in facts);assert n>0
    counts.update(current);macros.append((current['faithful']+current['less_precise'])/n);faithfulmacros.append(current['faithful']/n)
    removed=[f for f in c['fact_assessments'] if (r['id'],f['fact_id']) in excluded]
    if removed:affected.append({'position':r['position'],'id':r['id'],'candidate':c['candidate'],'removed_statuses':{f['fact_id']:f['status'] for f in removed},'retained_source_error':c['source_error'],'remaining_fact_missing':current['missing'],'remaining_fact_contradicted':current['contradicted'],'remaining_fact_unjudgeable':current['unjudgeable'],'remaining_fact_less_precise':current['less_precise'],'original_precision_issue':c['precision_issue'],'original_unjudgeable':c['unjudgeable']})
   total=sum(counts.values());assert total==plan[name+'_sensitivity_denominator']
   arms[arm]={'reports':40,'required_facts':total,'fact_statuses':dict(counts),'covered':counts['faithful']+counts['less_precise'],'coverage_micro_fraction':(counts['faithful']+counts['less_precise'])/total,'coverage_macro_fraction':sum(macros)/40,'faithful_only':counts['faithful'],'faithful_micro_fraction':counts['faithful']/total,'faithful_macro_fraction':sum(faithfulmacros)/40,'affected_cases':affected,'primary_full_faithful':primary['arms'][arm]['complete_faithful_chinese'],'affected_cases_all_remain_source_error':all(r['retained_source_error'] for r in affected)}
  result[name]=arms
 d={'purpose':'posthoc_annotation_scope_sensitivity_not_replacement_for_frozen_primary','human_reviews':0,'primary_preserved':True,'plan_sha256':hashlib.sha256((public/'source_annotation_sensitivity_plan_extended.json').read_bytes()).hexdigest(),'primary_sha256':hashlib.sha256((public/'final_ai_comparison.json').read_bytes()).hexdigest(),'results':result,'report_level_adjudication_needed':any(not a['affected_cases_all_remain_source_error'] for v in result.values() for a in v.values())}
 dest.write_text(json.dumps(d,indent=2)+'\n')
 for name,arms in result.items():
  print(name,{a:{k:v[k] for k in ['required_facts','covered','faithful_only','affected_cases_all_remain_source_error']} for a,v in arms.items()})
if __name__=='__main__':main()
