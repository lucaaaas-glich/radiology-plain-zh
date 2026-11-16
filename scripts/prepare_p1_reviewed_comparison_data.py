"""Prospective amended selection: frozen clear signal AND case audit support.

This combination is exploratory: the preceding 30-case validation evaluates the
original screen, not this post-validation policy. Prior proposals are preserved.
"""
import hashlib,json
from collections import Counter
from pathlib import Path
from radplain.common import fingerprint,read_jsonl,write_json,write_jsonl
from radplain.quality_p0 import features
from scripts.prepare_p1_datasets import choose_arms,common_support_summary,representative,stratum

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 root=Path.cwd();auditpath=root/'results/p1/comparison-20261117/clear22_audit.json';audit=json.loads(auditpath.read_text())
 old=root/'data/p1/comparison-20261117';out=old/'prepared-reviewed';summarypath=root/'results/p1/comparison-20261117/reviewed_data_selection.json'
 if out.exists() or summarypath.exists():raise FileExistsError('Preserve amended selection')
 screenpath=old/'screening/all_targets.local.jsonl';screen={r['id']:r for r in read_jsonl(screenpath)}
 clear={i for i,v in screen.items() if v['teacher_accepted'] and v['status']=='clear_issue'}
 cases={c['id']:c for c in audit['cases']}
 if set(cases)!=clear:raise ValueError('Audit must cover every clear flag')
 accepted={'supported','false_alarm','unjudgeable','mixed'}
 if any(c['assessment'] not in accepted for c in cases.values()):raise ValueError('Unknown audit label')
 supported={i for i,c in cases.items() if c['assessment']=='supported'}
 rows=read_jsonl(root/'data/pilot_v1/train.zh.jsonl');dev=read_jsonl(old/'prepared/S0/validation.zh.jsonl')
 groups=json.loads((root/'data/p0/pre-gpu-20261116/grouping/groups.json').read_text())['groups']
 original=json.loads((root/'results/p1/comparison-20261117/data_selection.json').read_text())
 blockedpaths=original['blocked_source_files_sha256']
 for p,s in blockedpaths.items():
  if digest(root/p)!=s:raise ValueError('Blocked inputs changed')
 blocked={groups[fingerprint(r['report'])] for path in blockedpaths for r in read_jsonl(root/path)}
 risks={r['id']:{'features':features(r['report'],r['source']),'screen_status':'risk_flagged' if r['id'] in supported else 'no_known_rule_flag'} for r in rows}
 for row in rows:
  if row['id'] in cases:
   c=cases[row['id']]
   if (digest_text(row['report'])!=c['source_sha256'] or digest_text(row['reference_zh'])!=c['target_sha256']):raise ValueError('Audit text binding mismatch')
 eligible=[r for r in rows if r['teacher_accepted'] and groups[r['id']] not in blocked]
 cfg=json.loads((old/'prepared/sampling_config.json').read_text());cfg['status']='exploratory_prospectively_amended_AI_reviewed_risk_selection'
 cfg['s1']='exclude frozen clear_issue only if complete22 project-AI audit supported; retain false/mixed/uncertain and all review_needed; unchanged targets'
 candidates=representative(eligible,groups,cfg['seed']);arms,strata=choose_arms(candidates,risks,cfg)
 if {r['id'] for r in arms['S0']}=={r['id'] for r in arms['S1']}:raise ValueError('No intervention; avoid duplicate training')
 out.mkdir()
 for arm,selected in arms.items():
  assert len(selected)>=512 and len({groups[r['id']] for r in selected})==len(selected)
  assert not {groups[r['id']] for r in selected}&{groups[r['id']] for r in dev}
  if arm=='S1':assert not {r['id'] for r in selected}&supported
  write_jsonl(out/arm/'train.zh.jsonl',selected);write_jsonl(out/arm/'validation.zh.jsonl',dev)
 assert Counter(stratum(r,risks[r['id']],cfg) for r in arms['S0'])==Counter(stratum(r,risks[r['id']],cfg) for r in arms['S1'])
 write_json(out/'sampling_config.json',cfg)
 summary={'purpose':cfg['status'],'human_reviews':0,'targets_rewritten':False,'independent_validation_of_combined_policy':False,
 'binding_sha256':{str(p.relative_to(root)):digest(p) for p in [auditpath,screenpath,root/'results/p1/comparison-20261117/data_selection.json',root/'data/pilot_v1/train.zh.jsonl']},
 'clear_audit_counts':dict(Counter(c['assessment'] for c in cases.values())),'eligible_unique_groups':len(candidates),'supported_exclusion_ids':sorted(supported),
 'shared_training_reports':len({r['id'] for r in arms['S0']}&{r['id'] for r in arms['S1']}),'same_joint_strata':True,'matched_strata':[{'features':list(k),'per_arm':n} for k,n in sorted(strata.items()) if n],
 'common_support':common_support_summary(candidates,risks,cfg),
 'arms':{arm:{'reports':len(v),'supported_flag_rows':sum(r['id'] in supported for r in v),'source_chars':sum(len(r['report']) for r in v),'target_chars':sum(len(r['reference_zh']) for r in v),'ids':[r['id'] for r in v]} for arm,v in arms.items()},
 'files_sha256':{str(p.relative_to(root)):digest(p) for p in out.rglob('*.jsonl')},'development_reports':len(dev)}
 write_json(summarypath,summary);print(json.dumps({k:summary[k] for k in ['clear_audit_counts','shared_training_reports','eligible_unique_groups']},ensure_ascii=False))
 for part in ('development20','final40'):
  path=root/f'examples/p1/{part}_20261117.lock.json';lock=json.loads(path.read_text());lock['training_inputs_sha256']={a:digest(out/a/'train.zh.jsonl') for a in arms}
  write_json(root/f'examples/p1/{part}_reviewed_20261117.lock.json',lock)
def digest_text(t):return hashlib.sha256(t.encode()).hexdigest()
if __name__=='__main__':main()
