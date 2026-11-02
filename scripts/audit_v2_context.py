"""Development ablation: empty prior context, all other dev7 prompts unchanged.

Does not mutate radplain or use any holdout. Keep the overlay script's hash with
its outputs. Selection must consider errors, coverage, readability and cost.
"""
import hashlib
import json
from pathlib import Path
import mlx.core as mx
import radplain.explain_v2 as pipeline
from radplain.common import read_jsonl, write_json
from radplain.mlx_backend import MLXBackend


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    previous_path=Path('results/v2/real-data/dev_thinking_critic_01/summary.json')
    previous=json.loads(previous_path.read_text())
    if previous['source_changed_during_run']:
        raise ValueError('Preceding experiment changed source')
    for file,sha in previous['source_files_sha256'].items():
        if digest(file)!=sha:
            raise ValueError('Code changed since reasoning diagnostic: '+file)
    inputs=[('synthetic','examples/challenge.jsonl','results/v2/dev_qwen35_contextless_01'),
            ('real','data/v2/real/dev.jsonl','results/v2/real-data/dev_qwen35_contextless_01')]
    if any(Path(out).exists() for _,_,out in inputs):
        raise ValueError('Do not overwrite previous experiment')
    source_files=[*Path('radplain').glob('*.py'),Path(__file__)]
    hashes={str(p):digest(p) for p in source_files}
    original=pipeline.messages_for
    def no_context(source,context,feedback=None):
        return original(source,'',feedback)
    pipeline.messages_for=no_context
    backend=MLXBackend('artifacts/v2/qwen35-4b')
    for split,source,out in inputs:
        folder=Path(out);folder.mkdir(parents=True,exist_ok=False)
        count=0;seconds=0
        with (folder/'outputs.jsonl').open('x') as f:
            for row in read_jsonl(source):
                result=pipeline.explain(row['report'],backend,max_tokens=384,use_critic=True)
                result['experimental_context_policy']='empty_previous_context_dev_ablation'
                f.write(json.dumps({'id':row['id'],'report':row['report'],'result':result},ensure_ascii=False)+'\n');f.flush()
                count+=1;seconds+=result['seconds']
                print(split,row['id'],round(result['seconds'],2),flush=True)
        write_json(folder/'runtime.json',{'count':count,'split':split,'stage':'development_only',
            'mode':'evidence-critic','context_policy':'empty_previous_context',
            'model':'artifacts/v2/qwen35-4b','max_tokens':384,
            'load_seconds':backend.load_seconds,'generation_seconds':seconds,
            'peak_mlx_allocated_bytes':mx.get_peak_memory(),
            'input_sha256':digest(source),'source_files_sha256':hashes,
            'source_changed_during_run':any(digest(p)!=sha for p,sha in hashes.items())})
    print('CONTEXT_ABLATION_COMPLETE',flush=True)


if __name__=='__main__':main()
