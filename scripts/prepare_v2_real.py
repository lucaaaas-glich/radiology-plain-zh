"""Freeze unused public reports by input length; no model outputs are read."""
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import pyarrow.parquet as pq
from radplain.common import fingerprint,read_jsonl,write_json,write_jsonl


def main():
    source=Path('data/v2/source/data/validation-00000-of-00001.parquet')
    out=Path('data/v2/real')
    out.mkdir(parents=True,exist_ok=False)
    seen={fingerprint(r['report']) for split in ('train','validation','test')
          for r in read_jsonl(f'data/pilot_v1/{split}.jsonl')}
    excluded=set(seen)
    pools=[[],[],[]]
    for r in pq.read_table(source).to_pylist():
        report=str(r.get('radiology_report') or '').strip()
        reference=str(r.get('layman_report') or '').strip()
        if not 20<=len(report)<=2400 or not reference:
            continue
        key=fingerprint(report)
        if key in seen:
            continue
        seen.add(key)
        index=0 if len(report)<400 else (1 if len(report)<1000 else 2)
        pools[index].append({'id':'real_'+key[:16],'report':report,'source_sha256':key,
                             'length_chars':len(report),'length_stratum':index})
    rng=random.Random(37102602)
    dev,test=[],[]
    for pool,n_dev in zip(pools,(4,6,6)):
        rng.shuffle(pool)
        assert len(pool)>=n_dev+8
        dev.extend(pool[:n_dev]);test.extend(pool[n_dev:n_dev+8])
    assert not {r['source_sha256'] for r in dev}&{r['source_sha256'] for r in test}
    assert not excluded&{r['source_sha256'] for r in dev+test}
    rng.shuffle(dev);rng.shuffle(test)
    write_jsonl(out/'dev.jsonl',dev)
    write_jsonl(out/'holdout.jsonl',test)
    write_json('examples/v2/real_manifest.json',{
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'outputs_seen':False,
        'dataset':'BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track',
        'revision':'6783ab38f9de28c4372799346f51d3927ae7eb9a','original_split':'validation',
        'source_parquet_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'seed':37102602,'remaining_pools_by_length':[len(x) for x in pools],
        'v1_unique_reports_excluded':len(excluded),
        'splits':{name:{'count':len(rows),'file_sha256':hashlib.sha256((out/f'{name}.jsonl').read_bytes()).hexdigest(),
                        'ids':[r['id'] for r in rows]} for name,rows in [('dev',dev),('holdout',test)]},
        'note':'Public reports; no clinician Chinese labels. Text deduplication is not patient independence. Raw text not redistributed.'})
    print('Frozen:',len(dev),'development,',len(test),'held-out; pools',[len(x) for x in pools])


if __name__=='__main__':
    main()
