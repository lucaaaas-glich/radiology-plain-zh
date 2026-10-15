import hashlib,json,time
from pathlib import Path
from huggingface_hub import snapshot_download,hf_hub_download,HfApi

root=Path(__file__).resolve().parents[1]
items=[('mlx-community/Qwen3-4B-Instruct-2507-4bit','50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b','qwen3-4b','Qwen/Qwen3-4B-Instruct-2507'),('mlx-community/Qwen3.5-4B-4bit','0e7ffd5c629ef7719d4cbc04069232580bfa9d9c','qwen35-4b','Qwen/Qwen3.5-4B')]
records=[]
for name,revision,slug,upstream in items:
 out=root/'artifacts/v2'/slug
 print('Downloading',name,revision,flush=True)
 start=time.time()
 snapshot_download(name,revision=revision,local_dir=out,allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.model','README.md','LICENSE*'],max_workers=2)
 up=HfApi().model_info(upstream)
 provenance=out/'upstream-provenance'
 for f in ('README.md','LICENSE'):
  hf_hub_download(upstream,f,revision=up.sha,local_dir=provenance)
 files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in out.glob('*') if p.is_file()}
 records.append({'repo':name,'revision':revision,'upstream':upstream,'upstream_revision_at_retrieval':up.sha,'note':'Upstream revision at retrieval is not proof of the converter original source revision. Community quantization, not trained by this project.','local_path':str(out.relative_to(root)),'download_seconds':time.time()-start,'files':files})
 (root/'results/v2/models.lock.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
 print('READY',slug,'seconds',round(time.time()-start,1),flush=True)
