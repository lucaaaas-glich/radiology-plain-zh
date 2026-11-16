"""One prospective evaluation of the already frozen conservative screen."""
import hashlib,json
from pathlib import Path
from collections import Counter
from radplain.common import read_jsonl,write_json,write_jsonl
from radplain.quality_p1 import screen_pair
from radplain.quality_p0 import risk_details

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 root=Path.cwd(); lockpath=root/'examples/p1/quality_candidate_20261117.lock.json'
 lock=json.loads(lockpath.read_text())
 for p,s in lock['files_sha256'].items():
  if sha(root/p)!=s:raise ValueError('Frozen candidate changed: '+p)
 raw=root/'data/p1/quality-validation-20261117/frozen_candidate_results.local.json'
 public=root/'results/p1/quality-validation-20261117/screen_counts.json'
 if raw.exists() or public.exists():raise FileExistsError('Retain original validation')
 synthpath=root/'examples/p1/quality_validation_20261117.jsonl'; realpath=root/'data/p1/quality-validation-20261117/real_packet.jsonl'
 protocol=json.loads((root/'examples/p1/quality_validation_protocol_20261117.json').read_text());manifest=json.loads((root/'examples/p1/real_quality_validation_manifest_20261117.json').read_text())
 assert sha(synthpath)==protocol['synthetic']['new_pairs_sha256'] and sha(realpath)==manifest['packet_sha256']
 synth=[];real=[]
 for pair in read_jsonl(synthpath):
  for kind in ('control','perturbed'):
   target=pair[kind+'_target'];r=screen_pair(pair['source'],target)
   synth.append({'id':pair['pair_id'],'kind':kind,'category':pair['category'],'old_flags':[i['code'] for i in risk_details(pair['source'],target)],**r})
 for row in read_jsonl(realpath):
  real.append({'id':row['id'],'old_flags':[i['code'] for i in risk_details(row['report'],row['reference_zh'])],**screen_pair(row['report'],row['reference_zh'])})
 write_json(raw,{'synthetic':synth,'real':real})
 summary={'purpose':'prospective_screen_validation_not_medical_accuracy','human_reviews':0,'candidate_lock_sha256':sha(lockpath),'inputs_sha256':{str(p.relative_to(root)):sha(p) for p in [synthpath,realpath]},'raw_result_sha256':sha(raw),
 'synthetic':{kind:{'n':30,'new_status_counts':dict(Counter(r['status'] for r in synth if r['kind']==kind)),'old_any_flag':sum(bool(r['old_flags']) for r in synth if r['kind']==kind)} for kind in ['control','perturbed']},
 'real_status_counts':dict(Counter(r['status'] for r in real)),
 'real_clear_ids':[r['id'] for r in real if r['status']=='clear_issue'],
 'issue_adjudication_pending':True,'readiness_decision_pending':True,
 'rows':[{'id':r['id'],'kind':r.get('kind'),'status':r['status'],'clear_codes':[i['code'] for i in r['issues'] if i['level']=='clear_issue'],'review_codes':[i['code'] for i in r['issues'] if i['level']=='review_needed'],'old_flags':r['old_flags']} for r in synth+real]}
 write_json(public,summary)
 print(json.dumps({k:summary[k] for k in ['synthetic','real_status_counts','real_clear_ids']},ensure_ascii=False))
if __name__=='__main__':main()
