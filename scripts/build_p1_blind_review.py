"""Freeze blinded source/fact/output packets; no automatic quality labels."""
import json,hashlib,random
from pathlib import Path
from radplain.common import read_jsonl,write_json

def sha(t):return hashlib.sha256(t.encode()).hexdigest()
def main():
 root=Path.cwd();raw=root/'data/p1/comparison-20261117';out=raw/'blind-review'
 if out.exists():raise FileExistsError('Keep blind packet and mapping unchanged')
 facts=[]
 for part in ('01_14','15_27','28_40'):
  facts+=json.loads((raw/f'source-review/final40_{part}.json').read_text())
 sources=read_jsonl(raw/'evaluation/final40.jsonl');assert len(sources)==len(facts)==40
 byarm={a:{v['id']:v for v in read_jsonl(raw/f'downloaded/data/p1/comparison-20261117-final40-{a}/predictions.jsonl')} for a in ['base','S0','S1']}
 if any(set(v)!={s['id'] for s in sources} for v in byarm.values()):raise ValueError('All fixed outputs required, including failed rows')
 packets=[];mapping=[]
 for n,(s,f) in enumerate(zip(sources,facts),1):
  assert s['id']==f['id'] and sha(s['report'])==f['source_sha256'] and not f['candidate_outputs_seen']
  for v in f['required_facts']:assert s['report'][v['start']:v['end']]==v['source_evidence']
  order=['base','S0','S1'];random.Random(3710261117+10000+n).shuffle(order);candidates=[]
  for label,arm in zip(['A','B','C'],order):
   r=byarm[arm][s['id']];assert r['report']==s['report']
   candidates.append({'candidate':label,'prediction':r['prediction'],'prediction_sha256':sha(r['prediction']),'generation_completed':r['status']=='completed','error_code':r['error_code'],'eos_completed':r['generation']['eos_completed']})
   mapping.append({'position':n,'id':s['id'],'candidate':label,'arm':arm,'prediction_sha256':sha(r['prediction'])})
  packets.append({'position':n,'id':s['id'],'report':s['report'],'source_sha256':sha(s['report']),'source_facts':f,'candidates':candidates})
 out.mkdir()
 write_json(out/'arm_mapping.private.json',mapping)
 for lo,hi in [(1,14),(15,27),(28,40)]:write_json(out/f'packet_{lo:02}_{hi:02}.json',packets[lo-1:hi])
 write_json(root/'results/p1/comparison-20261117/blind_packet_receipt.json',{'human_reviews':0,'blinding':'candidate letters independently shuffled per report; mapping excluded from reviewers; not independent clinical judgment','files_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.json')}})
if __name__=='__main__':main()
