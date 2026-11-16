"""Select source-only diagnostic evaluation before model outputs exist."""
import hashlib,json,random,re
from pathlib import Path
from radplain.common import fingerprint,read_jsonl,write_json,write_jsonl

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 import pyarrow.parquet as pq
 root=Path.cwd();out=root/'data/p1/comparison-20261117/evaluation'
 public=root/'examples/p1/comparison_sources_20261117.json'
 if out.exists() or public.exists():raise FileExistsError('Evaluation selection already exists')
 config_path=root/'configs/p1_comparison_protocol_20261117.json'
 config=json.loads(config_path.read_text());seed=config['sampling_seed']
 group_path=root/'data/p0/pre-gpu-20261116/grouping/groups.json'
 grouped=json.loads(group_path.read_text());groups=grouped['groups'];excluded=set(grouped['excluded_for_eval_hashes'])
 parquet=root/'data/v2/source/data/validation-00000-of-00001.parquet'
 raw=pq.read_table(parquet,columns=['source','radiology_report']).to_pylist()
 candidates={}
 for row in raw:
  report=row['radiology_report'];key=fingerprint(report)
  if key in excluded:continue
  candidates.setdefault(key,{'id':key,'report':report,'source':row['source'],'group':groups[key],
                            'length':len(re.sub(r'\s+',' ',report.casefold()).strip())})
 chosen=[];used=set()
 for offset,quota in enumerate(config['final_sampling']['normalized_char_length_by_source_quotas']):
  pool=sorted((r for r in candidates.values() if r['source']==quota['source'] and quota['min']<=r['length']<=quota['max']),key=lambda r:r['id'])
  random.Random(seed+offset).shuffle(pool);selected=[]
  for row in pool:
   if row['group'] in used:continue
   selected.append(row);used.add(row['group'])
   if len(selected)==quota['n']:break
  if len(selected)!=quota['n']:raise ValueError('Insufficient source quota')
  chosen.extend(selected)
 assert len(chosen)==40 and len(used)==40
 random.Random(seed+900).shuffle(chosen)
 dev_path=root/'data/p0/pre-gpu-20261116/prepared/S0/validation.zh.jsonl'
 dev=[{'id':r['id'],'report':r['report'],'source':r['source'],'group':groups[r['id']],'length':len(r['report'])} for r in read_jsonl(dev_path)[:20]]
 assert not ({r['group'] for r in dev}&used)
 out.mkdir(parents=True)
 manifests={}
 for name,rows in [('development20',dev),('final40',chosen)]:
  sources=[{'id':r['id'],'report':r['report']} for r in rows]
  path=out/(name+'.jsonl');write_jsonl(path,sources);path.chmod(0o600)
  manifests[name]={'path':str(path.relative_to(root)),'source_file_sha256':digest(path),
   'rows':[{'id':r['id'],'source_sha256':hashlib.sha256(r['report'].encode()).hexdigest(),
            'group':r['group'],'source':r['source'],'normalized_length':r['length']} for r in rows]}
 record={'purpose':config['purpose'],'protocol_sha256':digest(config_path),'source_pool_sha256':digest(parquet),
         'grouping_sha256':digest(group_path),'development_file_sha256':digest(dev_path),'manifests':manifests,
         'model_predictions_seen':False,'final_source_reference_or_teacher_target_read':False,'human_reviews':0}
 write_json(public,record)
 print(json.dumps({'development_reports':len(dev),'new_final_reports':len(chosen),'distinct_final_groups':len(used),'final_predictions_seen':False}))
if __name__=='__main__':main()
