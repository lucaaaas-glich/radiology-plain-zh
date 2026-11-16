"""Run a frozen risk policy and prepare matched, unchanged-label SFT conditions."""
import hashlib,json
from collections import Counter
from pathlib import Path
from radplain.common import fingerprint,read_jsonl,write_json,write_jsonl
from radplain.quality_p0 import features
from radplain.quality_p1 import screen_row
from scripts.prepare_p1_datasets import choose_arms,common_support_summary,representative,stratum

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 root=Path.cwd();lock_path=root/'examples/p1/quality_candidate_20261117.lock.json'
 lock=json.loads(lock_path.read_text())
 for path,sha in lock['files_sha256'].items():
  if digest(root/path)!=sha:raise ValueError('Frozen candidate changed: '+path)
 out=root/'data/p1/comparison-20261117/prepared';screen=root/'data/p1/comparison-20261117/screening'
 summary_path=root/'results/p1/comparison-20261117/data_selection.json'
 if any(p.exists() for p in (out,screen,summary_path)):raise FileExistsError('Cannot overwrite comparison data')
 rows=read_jsonl(root/'data/pilot_v1/train.zh.jsonl');dev=read_jsonl(root/'data/p0/pre-gpu-20261116/prepared/S0/validation.zh.jsonl')
 grouping_path=root/'data/p0/pre-gpu-20261116/grouping/groups.json';groups=json.loads(grouping_path.read_text())['groups']
 blocked_paths=['data/pilot_v1/validation.jsonl','data/pilot_v1/test.jsonl','data/v2/real/dev.jsonl','data/v2/real/holdout.jsonl','data/v3/real/holdout.jsonl','examples/challenge.jsonl','examples/v2/holdout.jsonl','data/p1/comparison-20261117/evaluation/final40.jsonl']
 blocked={groups[fingerprint(r['report'])] for path in blocked_paths for r in read_jsonl(root/path)}
 results=[];risks={};eligible=[];excluded=[]
 for row in rows:
  if row['id']!=fingerprint(row['report']):raise ValueError('Training ID/source mismatch')
  result=screen_row(row);results.append(result)
  risks[row['id']]={'features':features(row['report'],row['source']),
   'screen_status':'risk_flagged' if result['status']=='clear_issue' else 'no_known_rule_flag'}
  reason=('format_rejected' if not row['teacher_accepted'] else 'prior_or_new_evaluation_text_group' if groups[row['id']] in blocked else None)
  if reason:excluded.append({'id':row['id'],'reason':reason})
  else:eligible.append(row)
 cfg=json.loads((root/'configs/p1_data_selection.json').read_text());cfg['seed']=3710261117
 cfg['status']='controlled_AI_audited_risk_selection_not_clinical_quality_certification'
 cfg['s0']='common-support stratified random original targets'
 cfg['s1']='same strata, exclude clear_issue only; retain review_needed; unchanged targets'
 candidates=representative(eligible,groups,cfg['seed']);arms,strata=choose_arms(candidates,risks,cfg)
 if {r['id'] for r in arms['S0']}=={r['id'] for r in arms['S1']}:raise ValueError('No effective intervention; refuse duplicate-condition training')
 screened={r['id']:r for r in results}
 screen.mkdir(parents=True);out.mkdir()
 write_jsonl(screen/'all_targets.local.jsonl',results)
 for arm,selected in arms.items():
  write_jsonl(out/arm/'train.zh.jsonl',selected);write_jsonl(out/arm/'validation.zh.jsonl',dev)
  assert len({groups[r['id']] for r in selected})==len(selected)
  assert not ({groups[r['id']] for r in selected}&{groups[r['id']] for r in dev})
 write_json(out/'sampling_config.json',cfg)
 summary={'purpose':'controlled_source_consistency_comparison_AI_review_not_clinical','quality_candidate_lock_sha256':digest(lock_path),'human_reviews':0,
 'training_input_sha256':digest(root/'data/pilot_v1/train.zh.jsonl'),'grouping_sha256':digest(grouping_path),
 'blocked_source_files_sha256':{p:digest(root/p) for p in blocked_paths},
 'exclusions':dict(Counter(r['reason'] for r in excluded)),'eligible_unique_groups':len(candidates),
 'statuses_all_format_accepted':dict(Counter(r['status'] for r in results if r['teacher_accepted'])),
 'clear_issue_codes_format_accepted':dict(Counter(i['code'] for r in results if r['teacher_accepted'] for i in r['issues'] if i['level']=='clear_issue')),
 'common_support':common_support_summary(candidates,risks,cfg),'same_joint_strata':True,'matched_strata':[{'features':list(k),'per_arm':n} for k,n in sorted(strata.items()) if n],
 'shared_training_reports':len({r['id'] for r in arms['S0']}&{r['id'] for r in arms['S1']}),
 'arms':{arm:{'reports':len(selected),'source_chars':sum(len(r['report']) for r in selected),'target_chars':sum(len(r['reference_zh']) for r in selected),'screen_statuses':dict(Counter(screened[r['id']]['status'] for r in selected)),'ids':[r['id'] for r in selected]} for arm,selected in arms.items()},
 'development_reports':len(dev),'files_sha256':{str(p.relative_to(root)):digest(p) for p in out.rglob('*.jsonl')},
 'raw_screen_sha256':digest(screen/'all_targets.local.jsonl'),'targets_rewritten':False,'selection_not_claimed_all_correct':True}
 assert Counter(stratum(r,risks[r['id']],cfg) for r in arms['S0'])==Counter(stratum(r,risks[r['id']],cfg) for r in arms['S1'])
 write_json(summary_path,summary)
 print(json.dumps({k:summary[k] for k in ['statuses_all_format_accepted','exclusions','eligible_unique_groups','shared_training_reports']},ensure_ascii=False))
if __name__=='__main__':main()
