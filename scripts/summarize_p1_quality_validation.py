"""Join pre-authored AI labels to locked screen outputs without changing either."""
import hashlib,json
from collections import Counter
from pathlib import Path

def main():
 base=Path('results/p1/quality-validation-20261117');p=base/'prospective_result.json'
 if p.exists():raise FileExistsError(p)
 a=json.loads((base/'real_ai_review.json').read_text());s=json.loads((base/'screen_counts.json').read_text());r=json.loads(Path('data/p1/quality-validation-20261117/frozen_candidate_results.local.json').read_text())
 by={v['id']:v for v in r['real']};labels=a['cases'];assert set(by)=={v['id'] for v in labels}
 assert all(by[v['id']]['target_sha256']==v['target_sha256'] for v in labels)
 groups={}
 for key in ['observed_factual_error','required_information_missing','unjudgeable']:
  rows=[by[v['id']] for v in labels if v[key]];groups[key]={'n':len(rows),'status_counts':dict(Counter(x['status'] for x in rows))}
 realclear=[v for v in by.values() if v['status']=='clear_issue'];assert not realclear,'Issue-level adjudication required before false-clear count'
 d={'purpose':'prospective_screen_validation_not_clinical','human_reviews':0,'bindings_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [base/'real_ai_review.json',base/'screen_counts.json']},'synthetic':s['synthetic'],'real':{'cases':30,'status_counts':s['real_status_counts'],'project_AI_labels_overlap':groups,'observed_false_clear_flags':0,'clear_flags_total':0},'prespecified_false_clear_gate_passes':True,'interpretation':'The low-false-clear gate is vacuous for the real sample because zero real targets are clear-flagged. All 15 AI-observed factual-error targets remain eligible. This is NOT validation of meaningful data-quality improvement; a full clear-flag audit and prospective policy amendment precede GPU.','rules_retuned_after_validation':False}
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(json.dumps(d['real'],ensure_ascii=False))
if __name__=='__main__':main()
