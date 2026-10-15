"""Development-only comparison; never save or display the reasoning text."""
import hashlib
import json
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from radplain.mlx_backend import MLXBackend
from radplain.semantic_v2 import critique


class ThinkingCritic:
    def __init__(self, base, reasoning_budget=1024):
        self.base=base
        self.reasoning_budget=reasoning_budget

    def __call__(self, messages, max_tokens=512):
        tok=self.base.tokenizer
        prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
        input_count=len(tok.encode(prompt))
        if input_count>self.base.max_input_tokens:
            raise ValueError('Input limit exceeded; no truncation')
        start=time.perf_counter(); raw=''; last=None; reason=None
        stream=stream_generate(self.base.model,tok,prompt=prompt,
                 max_tokens=max_tokens+self.reasoning_budget,sampler=make_sampler(temp=0))
        try:
            for response in stream:
                raw+=response.text;last=response
                if '</think>' not in raw and response.generation_tokens>=self.reasoning_budget:
                    reason='reasoning_budget';break
                if '</think>' in raw:
                    answer=raw.rsplit('</think>',1)[-1].strip()
                    if len(tok.encode(answer,add_special_tokens=False))>max_tokens:
                        reason='length';break
        finally:
            stream.close()
        has_end='</think>' in raw
        answer=raw.rsplit('</think>',1)[-1].strip() if has_end else ''
        reasoning=raw.split('</think>',1)[0]
        finish=reason or getattr(last,'finish_reason',None)
        if not has_end:
            finish=reason or 'missing_think_end'
        result={'text':answer,'seconds':time.perf_counter()-start,'input_tokens':input_count,
                'generation_tokens':getattr(last,'generation_tokens',0),
                'reasoning_tokens':len(tok.encode(reasoning,add_special_tokens=False)),
                'finish_reason':finish,'thinking':True,'reasoning_budget':self.reasoning_budget}
        # Reasoning is deliberately discarded, including on failed/truncated runs.
        return result


def main():
    plan_path=Path('examples/v2/reasoning_diagnostic_manifest.json')
    plan=json.loads(plan_path.read_text())
    previous_path=Path('results/v2/real-data/dev_qwen35_critic_03/runtime.json')
    previous=json.loads(previous_path.read_text())
    if previous['count']!=16 or previous['source_changed_during_run']:
        raise ValueError('Preceding development run did not complete with unchanged source')
    for file,sha in previous['source_files_sha256'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=sha:
            raise ValueError('Do not change inference code between paired development experiments')
    data=Path('data/v2/thinking_dev_pairs.jsonl')
    if hashlib.sha256(data.read_bytes()).hexdigest()!=plan['pairs_sha256']:
        raise ValueError('Diagnostic data changed')
    rows=[json.loads(s) for s in data.read_text().splitlines()]
    out=Path('results/v2/real-data/dev_thinking_critic_01')
    out.mkdir(parents=True,exist_ok=False)
    files=[*Path('radplain').glob('*.py'),Path(__file__)]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    base=MLXBackend(plan['model']); thinking=ThinkingCritic(base,plan['max_reasoning_tokens'])
    results={}
    for mode,backend in [('direct',base),('thinking',thinking)]:
        records=[]
        with (out/(mode+'.jsonl')).open('x',encoding='utf-8') as stream:
            for row in rows:
                result=critique(row['source'],row['prediction'],backend,max_tokens=plan['max_verdict_tokens'])
                value=dict(row,result=result)
                records.append(value);stream.write(json.dumps(value,ensure_ascii=False)+'\n');stream.flush()
                print(mode,row['id'],result['verdict']['supported'],result['verdict']['valid'],round(result['generation']['seconds'],2),flush=True)
        results[mode]={'pairs':len(records),
          'bad_pairs_rejected':sum(r['expected_problem'] and not r['result']['verdict']['supported'] for r in records),
          'valid_pairs_rejected':sum(not r['expected_problem'] and not r['result']['verdict']['supported'] for r in records),
          'invalid_verdicts':sum(not r['result']['verdict']['valid'] for r in records),
          'seconds':sum(r['result']['generation']['seconds'] for r in records),
          'total_generated_tokens':sum(r['result']['generation']['generation_tokens'] for r in records)}
    result={'plan':plan,'arms':results,'load_seconds':base.load_seconds,
            'peak_mlx_allocated_bytes':mx.get_peak_memory(),'source_files_sha256':hashes,
            'source_changed_during_run':any(hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[str(p)] for p in files)}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('DEVELOPMENT_DIAGNOSTIC_COMPLETE',flush=True)


if __name__=='__main__':
    main()
