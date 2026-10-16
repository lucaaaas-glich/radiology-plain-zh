"""Plot completed SHA-bound V3 primary summaries; never fill missing results."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,default=Path('results/v3/final'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    files=[a.directory/(arm+'_summary.json') for arm in ('v2','v3')]
    summaries=[json.loads(f.read_text()) for f in files]
    if any(s['reports']!=20 or s['stage']!='final' for s in summaries):
        raise ValueError('Both full 20-report cohorts must be reviewed first')
    if summaries[0]['cohort_manifest_sha256']!=summaries[1]['cohort_manifest_sha256']:
        raise ValueError('Different cohorts')
    for suffix in ['.png','.svg','.json']:
        if a.out.with_suffix(suffix).exists():raise ValueError('Refusing to overwrite figure')
    font=FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
    colors=['#36745c','#c45f50','#dcae57','#8b99a5']
    keys=['complete_faithful','observed_error','incomplete_without_observed_error','precision_concern_without_clear_error']
    labels=['讲全且未发现错误','发现明确事实错误','未发现错误，但没有讲全','讲全但表达有疑问']
    fig,ax=plt.subplots(figsize=(10.4,4.25),layout='constrained')
    fig.patch.set_facecolor('#fafbf8');ax.set_facecolor('#fafbf8')
    for y,summary in enumerate(summaries):
        left=0
        for key,color in zip(keys,colors):
            value=summary['mutually_exclusive_outcomes'].get(key,0)
            ax.barh(y,value,left=left,height=.40,color=color)
            if value:ax.text(left+value/2,y,str(value),ha='center',va='center',color='white' if key in keys[:2] else '#20372e',fontsize=13,fontweight='bold')
            left+=value
        if left!=20:raise ValueError('Outcome categories do not sum to cohort size')
    ax.set_yticks([0,1],['V2 原流程','V3 事实清单流程'],fontproperties=font,fontsize=13)
    ax.invert_yaxis();ax.set_xlim(0,20);ax.set_xticks([0,5,10,15,20]);ax.set_xlabel('报告份数（本轮新选取的同一批 20 份真实长报告）',fontproperties=font,fontsize=11)
    ax.set_title('同时看有没有说错，也看有没有讲全',fontproperties=font,fontsize=19,loc='left',pad=25)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.tick_params(length=0,pad=8)
    ax.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(colors,labels)],loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2,frameon=False,prop=font)
    fig.suptitle('项目 AI 逐例复核；非医生评审、非医学准确率',fontproperties=font,fontsize=10,color='#526673',y=.99)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['.png','.svg']:fig.savefig(a.out.with_suffix(suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
    record={'source_summary_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
            'cohort_manifest_sha256':summaries[0]['cohort_manifest_sha256'],
            'not_clinical_accuracy':True,'outcomes':{arm:s['mutually_exclusive_outcomes'] for arm,s in zip(['v2','v3'],summaries)}}
    a.out.with_suffix('.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
