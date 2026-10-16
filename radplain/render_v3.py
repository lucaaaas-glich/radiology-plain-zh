"""Readable standalone output with optional source checklist, no failed drafts."""
from html import escape
from urllib.parse import urlparse

LABELS={'entity':'对象','location':'部位','quantity':'数值与单位','negation':'否定范围',
        'certainty':'确定程度','time_change':'时间与变化','condition_action':'条件与建议','qualifier':'其他限定'}


def public_result(result):
    return {k:([ {a:b for a,b in row.items() if a!='attempts'} for row in value ] if k=='segments' else value)
            for k,value in result.items() if k!='calls'}


def render_text(result):
    lines=[]
    for row in result['segments']:
        lines.append(f"[{row['id']}] 原文：{row['source']}")
        lines.append('中文：'+row['explanation_zh'] if row['status']=='generated'
                     else '章节标记，保留原样。' if row['status']=='structure' else row['reason'])
        lines.append('')
    lines.append(result['notice'])
    return '\n'.join(lines)


def render_html(result):
    cards=[]
    for row in result['segments']:
        if row['status']=='structure':
            cards.append('<h2 class="section">'+escape(row['source'])+'</h2>')
            continue
        generated=row['status']=='generated'
        content=row['explanation_zh'] if generated else row['reason']
        facts=[]
        for fact in (row.get('ledger') or {}).get('facts',[]):
            details=''.join('<li><strong>'+LABELS[k]+'：</strong>'+escape(v)+'</li>'
                            for k,v in fact.items() if k in LABELS and v)
            facts.append('<div class="fact"><blockquote>'+escape(fact['evidence'])+'</blockquote><ul>'+details+'</ul></div>')
        checklist=('<details><summary>查看模型整理的原文事实清单</summary><p class="muted">以下是原文摘录；选择和归类仍可能出错。</p>'
                   +''.join(facts)+'</details>') if facts else ''
        cards.append('<article class="card '+('' if generated else 'fallback')+'"><small>'
                     +str(row['id'])+' · '+('中文解释' if generated else '本段未完成')+'</small><p class="explanation">'
                     +escape(content)+'</p><details '+('' if generated else 'open')+'><summary>对照原文</summary><blockquote>'
                     +escape(row['source'])+'</blockquote></details>'+checklist+'</article>')
    terms=[]
    for item in result.get('glossary',[]):
        url=item['url']
        link=('<a href="'+escape(url,quote=True)+'" rel="noopener noreferrer">词义来源</a>') if urlparse(url).scheme=='https' else ''
        terms.append('<li><strong>'+escape(item['term'])+'</strong>：'+escape(item['definition_zh'])+' '+link+'</li>')
    coverage=result['coverage']
    status=f"{coverage['generated_segments']} / {coverage['eligible_segments']} 个内容片段已生成中文；{coverage['source_only_segments']} 个片段保留原文。"
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V3 中文报告解释 · 原文事实清单</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f3;color:#24372e;font:17px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}main{max-width:900px;margin:auto;padding:40px 24px}h1{font-size:32px;line-height:1.3}.muted,small,footer{font-size:14px;color:#59695f}.status{background:#e4ede7;padding:16px;border-radius:10px}.card{background:white;border:1px solid #d8e3db;border-radius:12px;margin:16px 0;padding:22px}.fallback{background:#fff8e8;border-color:#e7d3a5}.explanation{font-size:19px;white-space:pre-wrap;overflow-wrap:anywhere}details{font-size:14px;margin:12px 0}summary{cursor:pointer;color:#28664e}blockquote{margin:10px 0;padding:12px 16px;background:#f0f4ef;border-left:3px solid #b4cbbd;white-space:pre-wrap;overflow-wrap:anywhere}.fact{border-bottom:1px solid #d8e3db}li{overflow-wrap:anywhere}.terms{padding-left:22px}.terms li{margin:12px 0}a{color:#28664e}.section{font-size:21px;margin-top:30px}footer{border-top:1px solid #d8e3db;margin-top:30px;padding-top:18px}@media print{body{background:white}.card{break-inside:avoid}main{padding:0}}@media(max-width:540px){main{padding:24px 16px}.card{padding:18px}h1{font-size:26px}}
</style></head><body><main><header><small>V3 · 研究演示</small><h1>看懂报告，也能对照原文</h1><p>先看中文，需要时展开原句和模型整理的事实清单。</p><div class="status">'''+escape(status)+'''<br><span class="muted">这是生成范围，不是医学准确率。</span></div></header>'''+''.join(cards)+('<section><h2>术语帮助</h2><p class="muted">只解释词义，不代表报告肯定发现了这些情况。</p><ul class="terms">'+''.join(terms)+'</ul></section>' if terms else '')+'<footer>'+escape(result['notice'])+'</footer></main></body></html>'
