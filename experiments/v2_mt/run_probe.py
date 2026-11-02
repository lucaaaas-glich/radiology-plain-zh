"""Development-only NLLB probe; does not touch the frozen V2 candidate."""
import hashlib
import importlib.metadata
import json
import resource
import time
from pathlib import Path

import ctranslate2
from transformers import AutoTokenizer
from radplain.guard_v2 import spans, administrative_instruction, review
from radplain.glossary_v2 import glossary
from radplain.explain_v2 import render_text


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    root=Path.cwd();plan_path=root/'experiments/v2_mt/probe_plan.json'
    plan=json.loads(plan_path.read_text());model=root/'artifacts/v2/nllb600-int8'
    model_lock=json.loads((model/'download_verified.json').read_text())
    for item in model_lock['files']:
        if sha(model/item['name'])!=item['sha256']:raise ValueError('Changed model file '+item['name'])
    if 'int8' not in ctranslate2.get_supported_compute_types('cpu'):
        raise RuntimeError('CPU int8 unavailable; no silent compute-type change')
    for path,meta in plan['inputs'].items():
        if sha(root/path)!=meta['sha256']:raise ValueError('Changed development input')
    source_paths=['experiments/v2_mt/run_probe.py','radplain/guard_v2.py','radplain/glossary_v2.py','radplain/explain_v2.py','radplain/quantity_v2.py']
    before={p:sha(root/p) for p in source_paths}
    load_start=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,src_lang='eng_Latn',trust_remote_code=False)
    if tokenizer.convert_tokens_to_ids('zho_Hans')==tokenizer.unk_token_id:raise ValueError('Missing target language token')
    translator=ctranslate2.Translator(str(model),device='cpu',compute_type='int8',inter_threads=1,intra_threads=2)
    load_seconds=time.perf_counter()-load_start;start=time.perf_counter()
    sources=[('synthetic','examples/challenge.jsonl','results/v2/dev_nllb600_probe'),('real','data/v2/real/dev.jsonl','results/v2/real-data/dev_nllb600_probe')]
    for split,input_path,output_path in sources:
        output=root/output_path
        output.mkdir(parents=True,exist_ok=False)
        rows=[json.loads(line) for line in (root/input_path).read_text().splitlines()]
        if split=='synthetic':rows=[r for r in rows if r['id'].endswith('_en')]
        durations=[]
        with (output/'outputs.jsonl').open('x') as f:
            for original in rows:
                begun=time.perf_counter();report=original['report'];segments=[]
                for i,piece in enumerate(spans(report),1):
                    if time.perf_counter()-start>plan['max_generation_seconds']:raise TimeoutError('900s generation budget reached')
                    row=dict(piece,id=i,explanation_zh=None,attempts=[])
                    if administrative_instruction(piece['source']):
                        row.update(status='excluded_instruction',reason='原文包含疑似行政指令，未作为影像发现解释。');segments.append(row);continue
                    ids=tokenizer.encode(piece['source'],truncation=False)
                    if len(ids)>512:
                        row.update(status='source_only',reason='超过本翻译探针512输入token限制，未截断原文。');segments.append(row);continue
                    tokens=tokenizer.convert_ids_to_tokens(ids)
                    if tokens[0]!='eng_Latn':raise ValueError('Unexpected NLLB source language prefix')
                    t=time.perf_counter()
                    translated=translator.translate_batch([tokens],target_prefix=[['zho_Hans']],beam_size=4,max_decoding_length=512,max_input_length=0)[0]
                    raw=translated.hypotheses[0]
                    if not raw or raw[0]!='zho_Hans':raise ValueError('Unexpected target language prefix')
                    text=tokenizer.decode(tokenizer.convert_tokens_to_ids(raw[1:]),skip_special_tokens=True).strip()
                    finish='length' if len(raw)>=512 else 'stop'
                    attempt={'text':text,'seconds':time.perf_counter()-t,'input_tokens':len(ids),'generation_tokens':len(raw),'finish_reason':finish,'finish_reason_note':'Length-bound check; CTranslate2 strips the ending token.','check':review(piece['source'],text,finish)}
                    row['attempts']=[attempt]
                    if text and finish=='stop':row.update(status='generated',explanation_zh=text)
                    else:row.update(status='source_only',reason='翻译为空或触及长度上限，未展示不完整中文。')
                    segments.append(row)
                eligible=[s for s in segments if s['status']!='excluded_instruction'];generated=sum(s['status']=='generated' for s in eligible)
                result={'version':'nllb600-int8-development-probe','source_report':report,'use_critic':False,'segments':segments,'glossary':glossary('\n'.join(s['source'] for s in eligible)),'coverage':{'generated_segments':generated,'eligible_segments':len(eligible),'source_only_segments':len(eligible)-generated,'excluded_instruction_segments':len(segments)-len(eligible),'fully_generated':bool(eligible) and generated==len(eligible)},'seconds':time.perf_counter()-begun,'notice':'额外翻译研究探针；NLLB不是医学领域模型，没有临床验证。规则结果仅供分析，不参与筛选。'}
                result['display_text']=render_text(result);durations.append(result['seconds'])
                f.write(json.dumps({'id':original['id'],'report':report,'result':result},ensure_ascii=False)+'\n');f.flush()
                print(split,original['id'],round(result['seconds'],2),flush=True)
        after={p:sha(root/p) for p in source_paths}
        runtime={'stage':'development_only','split':split,'count':len(rows),'generation_seconds':sum(durations),'load_seconds':load_seconds,'model_repo':plan['model']['repo'],'model_revision':plan['model']['revision'],'model_files':model_lock,'input_sha256':sha(root/input_path),'plan_sha256':sha(plan_path),'device':'cpu','compute_type':translator.compute_type,'source_sha256':before,'source_changed_during_run':before!=after,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'dependencies':{x:importlib.metadata.version(x) for x in ['ctranslate2','transformers','tokenizers','sentencepiece','numpy','huggingface-hub']}}
        (output/'runtime.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2)+'\n')
        if before!=after:raise RuntimeError('Source changed during probe')


if __name__=='__main__':main()
