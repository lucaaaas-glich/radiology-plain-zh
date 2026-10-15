"""Plot mutually exclusive outcomes, including abstentions and review concerns.

Only validated case-level reviews determine facts. No guard pass-rate is used.
Run with matplotlib; inference dependencies are not imported.
"""
import argparse
import json
from pathlib import Path
from scripts.summarize_v2 import load_rows,validate,fully_generated


def buckets(rows,review):
    checked=validate(rows,review)
    counts={'faithful_complete':0,'observed_error':0,'incomplete_without_observed_error':0,'remaining_precision_concern':0}
    for row in rows:
        c=checked[row['id']]
        if c['errors']:key='observed_error'
        elif c['complete_faithful_chinese']:key='faithful_complete'
        elif not fully_generated(row['result']):key='incomplete_without_observed_error'
        else:key='remaining_precision_concern'
        counts[key]+=1
    assert sum(counts.values())==len(rows)
    return counts


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    plan=json.loads(a.manifest.read_text())
    if plan['stage']=='final':
        barrier=json.loads(Path('examples/v2/final_inspection_barrier.json').read_text())
        if barrier['state']!='released':
            raise ValueError('Final inspection barrier is still held')
    elif plan['stage']!='development':
        raise ValueError('Explicit development/final stage required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    local_font=Path('/System/Library/Fonts/STHeiti Light.ttc')
    if local_font.exists():
        font_manager.fontManager.addfont(str(local_font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(local_font)).get_name()
    else:
        plt.rcParams['font.family']=['Noto Sans CJK SC','DejaVu Sans']
    values=[];total=None
    for arm in plan['arms']:
        rows=load_rows(arm['outputs']);review=json.loads(Path(arm['review']).read_text())
        if total is None:total=len(rows);ids={r['id'] for r in rows}
        if len(rows)!=total or {r['id'] for r in rows}!=ids:
            raise ValueError('All plotted arms must contain identical case IDs')
        values.append(buckets(rows,review))
    fig,ax=plt.subplots(figsize=(10.6,4.8))
    fig.subplots_adjust(left=.27,right=.96,bottom=.32,top=.77)
    keys=['faithful_complete','observed_error','incomplete_without_observed_error','remaining_precision_concern']
    labels=['完整且忠实','发现事实错误','未见明确错误，但未完成','已生成，但仍有精度顾虑']
    colors=['#3d7b93','#cd7753','#b9c5cc','#d9bf76']
    left=[0]*len(values)
    for key,label,color in zip(keys,labels,colors):
        widths=[v[key] for v in values]
        bars=ax.barh(range(len(values)),widths,left=left,height=.62,color=color,label=label)
        for i,(bar,n) in enumerate(zip(bars,widths)):
            if n:ax.text(left[i]+n/2,i,str(n),ha='center',va='center',fontsize=11,color='#162f3a')
        left=[x+y for x,y in zip(left,widths)]
    ax.set_yticks(range(len(values)),[x['label'] for x in plan['arms']]);ax.invert_yaxis()
    ax.set_xlim(0,total);ax.set_xticks([x for x in range(total+1) if x%max(1,total//8)==0])
    ax.set_xlabel('报告份数（每份只归入一类）',labelpad=10)
    ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,pad=12)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle(plan['title'],fontsize=17,x=.055,ha='left',y=.95,color='#203b48')
    fig.text(.055,.865,plan['subtitle'],fontsize=10,color='#576d77')
    ax.legend(loc='upper left',bbox_to_anchor=(-.30,-.32),ncol=2,frameon=False,fontsize=10,columnspacing=2)
    fig.text(.055,.055,'项目AI逐例复核，未做医生验证。有事实错误的报告可能同时有回退；优先计入错误类。',fontsize=9,color='#576d77')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg']:
        file=a.out.with_suffix('.'+ext)
        if file.exists():raise ValueError('Refusing to overwrite '+str(file))
        fig.savefig(file,dpi=180,facecolor='#fbfcfd')
    plt.close(fig)
    print(json.dumps({'stage':plan['stage'],'reports_per_arm':total,'outcomes':values},ensure_ascii=False))


if __name__=='__main__':main()
