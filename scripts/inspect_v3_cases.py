"""Compact read-only review packets; never assigns clinical outcome labels."""
import argparse
import json
from pathlib import Path
from scripts.summarize_v2 import fingerprint, load_rows
from radplain.ledger_v3 import parse_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--outputs',type=Path,nargs='+',required=True)
    p.add_argument('--start',type=int,default=1)
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--ledger',action='store_true')
    p.add_argument('--ledger-extractions',action='store_true',
                   help='Also inspect original extracted fields, including rejected checklists')
    a=p.parse_args()
    arms=[(path.parent.name,{r['id']:r for r in load_rows(path)}) for path in a.outputs]
    order=list(arms[0][1])[a.start-1:a.start-1+a.count]
    for position,key in enumerate(order,a.start):
        print('\nCASE',position,key,'\nSOURCE:',arms[0][1][key]['report'])
        for label,rows in arms:
            if key not in rows:
                print(label,'PENDING');continue
            row=rows[key]
            print('\nARM',label,'SHA',fingerprint(row))
            print('SECONDS',row.get('generation_accounting',{}).get('seconds'))
            if 'error' in row:
                print('EXECUTION ERROR',row['error']);continue
            result=row['result'];print('COVERAGE',result.get('coverage',{}))
            for s in result.get('segments',[]):
                print(f"[{s['id']} {s.get('start')}:{s.get('end')}]",s['explanation_zh'] if s['status']=='generated' else s['status'].upper()+': '+s['source'])
                if a.ledger and s.get('ledger'):
                    print(' LEDGER',s['ledger']['role'],
                          [{k:v for k,v in fact.items() if k!='evidence'} for fact in s['ledger']['facts']])
            if a.ledger_extractions:
                print('EXTRACTION VALIDATION',json.dumps(result.get('ledger_batches',[]),ensure_ascii=False))
                for call in result.get('calls',[]):
                    if call.get('stage')!='ledger':continue
                    print('ORIGINAL EXTRACTION',call['ids'],'FINISH',call.get('finish_reason'))
                    try:
                        value=parse_json(call)
                        for unit in value['units']:
                            print(' UNIT',unit['id'],unit.get('role'))
                            for fact in unit['facts']:
                                fields={k:v for k,v in fact.items() if k!='evidence'}
                                print('  ',json.dumps(fields or fact,ensure_ascii=False))
                    except (ValueError,KeyError,TypeError):
                        # Do not silently omit malformed extractions from review.
                        print(call['text'])


if __name__=='__main__':main()
