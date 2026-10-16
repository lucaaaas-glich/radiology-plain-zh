"""Compact read-only review packets; never assigns clinical outcome labels."""
import argparse
import json
from pathlib import Path
from scripts.summarize_v2 import fingerprint, load_rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--outputs',type=Path,nargs='+',required=True)
    p.add_argument('--start',type=int,default=1)
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--ledger',action='store_true')
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


if __name__=='__main__':main()
