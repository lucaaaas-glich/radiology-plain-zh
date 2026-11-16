"""Freeze blinded source/fact/output packets; no automatic quality labels."""
import json,hashlib,random,secrets
from pathlib import Path
from radplain.common import read_jsonl,write_json
from radplain.evaluate_p1 import load_sources,read_journal
from radplain.verify_p1 import digest_json

def sha(t):return hashlib.sha256(t.encode()).hexdigest()
def main():
 root=Path.cwd();raw=root/'data/p1/comparison-20261117';out=raw/'blind-review'
 if out.exists():raise FileExistsError('Keep blind packet and mapping unchanged')
 factlock=json.loads((root/'examples/p1/final_source_facts_20261117.lock.json').read_text())
 for path,digest in factlock['source_review_files_sha256'].items():
  if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise ValueError('Pre-output source facts changed')
 facts=[]
 for part in ('01_14','15_27','28_40'):
  facts+=json.loads((raw/f'source-review/final40_{part}.json').read_text())
 sources=read_jsonl(raw/'evaluation/final40.jsonl');assert len(sources)==len(facts)==40
 byarm={}
 locked_sources,lock,hashes=load_sources(raw/'evaluation/final40.jsonl',root/'examples/p1/final40_reviewed_20261117.lock.json')
 for a in ['base','S0','S1']:
  folder=raw/f'downloaded/data/p1/comparison-20261117-final40-{a}'
  runlock=json.loads((folder/'run.lock.json').read_text());identity=runlock['identity']
  assert identity['arm']==a and identity['input_lock_sha256']==hashes['input_lock_sha256'] and digest_json(identity)==runlock['identity_sha256']
  rows=read_journal(folder/'predictions.jsonl',locked_sources,runlock['identity_sha256'])
  assert len(rows)==len({r['id'] for r in rows})==40
  byarm[a]={v['id']:v for v in rows}
 if any(set(v)!={s['id'] for s in sources} for v in byarm.values()):raise ValueError('All fixed outputs required, including failed rows')
 packets=[];mapping=[];private_seed=secrets.randbits(256);rng=random.Random(private_seed)
 for n,(s,f) in enumerate(zip(sources,facts),1):
  assert s['id']==f['id'] and sha(s['report'])==f['source_sha256'] and not f['candidate_outputs_seen']
  for v in f['required_facts']:assert s['report'][v['start']:v['end']]==v['source_evidence']
  order=['base','S0','S1'];rng.shuffle(order);candidates=[]
  for label,arm in zip(['A','B','C'],order):
   r=byarm[arm][s['id']];assert r['report']==s['report']
   candidates.append({'candidate':label,'prediction':r['prediction'],'prediction_sha256':sha(r['prediction']),'generation_completed':r['status']=='completed','error_code':r['error_code'],'eos_completed':r['generation']['eos_completed']})
   mapping.append({'position':n,'id':s['id'],'candidate':label,'arm':arm,'prediction_sha256':sha(r['prediction'])})
  packets.append({'position':n,'id':s['id'],'report':s['report'],'source_sha256':sha(s['report']),'source_facts':f,'candidates':candidates})
 assert len(mapping)==len({(m['id'],m['candidate']) for m in mapping})==120
 out.mkdir()
 write_json(out/'randomization_seed.private.json',{'seed':private_seed,'purpose':'Private blinding seed; do not share with candidate reviewers'})
 write_json(out/'arm_mapping.private.json',mapping)
 for lo,hi in [(1,14),(15,27),(28,40)]:write_json(out/f'packet_{lo:02}_{hi:02}.json',packets[lo-1:hi])
 write_json(root/'results/p1/comparison-20261117/blind_packet_receipt.json',{'human_reviews':0,'blinding':'candidate letters independently shuffled per report; mapping excluded from reviewers; not independent clinical judgment','files_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*out.glob('*.json'),root/'examples/p1/final_source_facts_20261117.lock.json',root/'examples/p1/final40_reviewed_20261117.lock.json',raw/'blind_review_instructions.txt']}})
if __name__=='__main__':main()
