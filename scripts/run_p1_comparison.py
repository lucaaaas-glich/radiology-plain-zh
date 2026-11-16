"""Execute one fixed B0/S0/S1 comparison; never select from final predictions."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

def require_complete_generation(public_output):
 manifests=sorted(Path(public_output).glob("manifest-*.json"))
 if not manifests:raise RuntimeError("Missing evaluation manifest")
 summary=json.loads(manifests[-1].read_text());c=summary["counts"]
 if c["pending"]!=0 or c["terminal"]!=c["requested"] or c["completed"]+c["failed"]!=c["terminal"]:
  raise RuntimeError("Incomplete evaluation: stop and recover only missing IDs")
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
 root=Path.cwd();dest=root/'results/p1/comparison-20261117-execution'
 if dest.exists():raise FileExistsError('Use a fresh experiment; no implicit training resume')
 dest.mkdir(parents=True);steps=[];start=time.time()
 def run(args,allow_terminal_failures=False):
  t=time.perf_counter(); r=subprocess.run([sys.executable,*args]);steps.append({'args':args,'exit_code':r.returncode,'elapsed_seconds':time.perf_counter()-t})
  if r.returncode and not(allow_terminal_failures and r.returncode==2):raise RuntimeError('Comparison step failed; inspect retained artifacts')
 try:
  for arm in ('S0','S1'):
   run(['-m','radplain.train_p1','train','--mode','sft','--config','configs/p1_sft_comparison_20261117.json','--data',f'data/p1/comparison-20261117/prepared-reviewed/{arm}','--tokenizer',a.model,'--model',a.model,'--out',f'artifacts/p1/comparison-20261117-{arm}'])
  selected={}
  for arm in ('S0','S1'):
   folder=root/f'artifacts/p1/comparison-20261117-{arm}';s=json.loads((folder/'selection.json').read_text());h=json.loads((folder/'history.json').read_text())
   best=min(h,key=lambda v:(v['validation_answer_token_loss'],v['epoch']))['epoch']
   if s['epoch']!=best or s['final_test_read'] is not False:raise ValueError('Checkpoint selection does not match frozen dev-loss rule')
   selected[arm]=best
  (dest/'checkpoint_selection.json').write_text(json.dumps(selected,indent=2)+'\n')
  # Both conditions completed and checkpoint choices fixed before any final generation.
  for split in ('development20','final40'):
   for arm in ('base','S0','S1'):
    args=['-m','radplain.evaluate_p1','--arm',arm,'--sources',f'data/p1/comparison-20261117/evaluation/{split}.jsonl','--input-lock',f'examples/p1/{split}_reviewed_20261117.lock.json','--model',a.model,'--out',f'data/p1/comparison-20261117-{split}-{arm}','--public-out',f'results/p1/comparison-20261117-{split}-{arm}']
    if arm!='base':args+=['--adapter',f'artifacts/p1/comparison-20261117-{arm}/epoch-{selected[arm]}','--epoch',str(selected[arm])]
    run(args,True)
    require_complete_generation(root/f'results/p1/comparison-20261117-{split}-{arm}')
  state='finished_all_fixed_commands'
 except BaseException as e:
  state='failed';raise
 finally:
  (dest/'execution.json').write_text(json.dumps({'state':state,'steps':steps,'elapsed_wall_seconds':time.time()-start,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'final_quality_selection':False},indent=2)+'\n')
if __name__=='__main__':main()
