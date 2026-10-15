"""A standalone readable report: source toggles, no scripts or external assets."""
from html import escape
from urllib.parse import urlparse


def render_html(result):
    coverage=result['coverage']
    items=[]
    for row in result['segments']:
        generated=row['status']=='generated'
        content=row['explanation_zh'] if generated else row['reason']
        kind='解释' if generated else '保留原文'
        items.append(f'''<article class="card {'fallback' if not generated else ''}">
        <div class="eyebrow">{row['id']:02d} · {kind}</div>
        <p class="explanation">{escape(content)}</p>
        <details {'open' if not generated else ''}><summary>查看对应原文</summary>
        <blockquote>{escape(row['source'])}</blockquote></details></article>''')
    terms=[]
    for entry in result['glossary']:
        url=entry['url']
        link=f'<a href="{escape(url,quote=True)}" target="_blank" rel="noopener noreferrer">词义来源 ↗</a>' if urlparse(url).scheme=='https' else ''
        terms.append(f'<div class="term"><h3>{escape(entry["term"])}</h3><p>{escape(entry["definition_zh"])}</p>{link}</div>')
    completed=coverage['generated_segments']
    total=coverage['eligible_segments']
    status=f'{completed} / {total} 个片段提供了中文解释'
    if coverage['source_only_segments']:
        status+=f' · {coverage["source_only_segments"]} 个片段保留原文待核对'
    glossary_section=''
    if terms:
        glossary_section='<section class="glossary"><h2>把词语说清楚</h2><p class="muted">以下仅解释词义，是否存在该发现以报告为准。</p><div class="terms">'+''.join(terms)+'</div></section>'
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>中文影像报告解释</title><style>
:root{color-scheme:light;--ink:#20332f;--muted:#5c6d67;--line:#d9e3dc;--accent:#256652}
*{box-sizing:border-box}body{margin:0;background:#f3f5ef;color:var(--ink);font:17px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC","Noto Sans CJK SC",sans-serif}
main{max-width:860px;margin:auto;padding:48px 24px 64px}header{margin-bottom:30px}.tag{font-size:12px;letter-spacing:.1em;color:var(--accent)}h1{font-size:clamp(27px,5vw,40px);line-height:1.35;margin:12px 0}h2{font-size:24px;margin:0 0 4px}h3{font-size:17px;margin:0 0 6px}
.muted,.status{color:var(--muted);font-size:14px}.status{border-top:1px solid var(--line);padding-top:16px;margin-top:24px}.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:22px 26px;margin:14px 0}.eyebrow{font-size:12px;letter-spacing:.06em;color:var(--accent)}.explanation{font-size:19px;margin:10px 0 16px;white-space:pre-wrap;overflow-wrap:anywhere}
details{font-size:14px}summary{cursor:pointer;color:var(--accent)}blockquote{margin:10px 0 0;padding:14px 18px;background:#f6f8f5;border-left:3px solid var(--line);color:#4a5a54;white-space:pre-wrap;overflow-wrap:anywhere}.fallback{background:#fffaf0;border-color:#e5d5aa}.fallback .eyebrow{color:#8c6223}.glossary{margin-top:40px}.terms{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:20px}.term{padding:20px;border-top:2px solid var(--line)}.term p{font-size:15px;margin:0 0 10px}a{font-size:12px;color:var(--accent)}footer{margin-top:36px;border-top:1px solid var(--line);padding-top:20px;font-size:13px;color:var(--muted)}
@media(max-width:540px){main{padding:28px 16px}.card{padding:18px}.terms{grid-template-columns:1fr}.explanation{font-size:18px}}@media print{body{background:white}.card{break-inside:avoid}details{display:block}summary{display:none}details:not([open])>blockquote{display:block}.terms{display:block}.term{break-inside:avoid}main{padding:0}a{color:inherit}}
</style><main><header><div class="tag">本机生成 · 研究演示</div><h1>影像报告，读懂每一句</h1>
<p class="muted">每段解释都对应一段原文。点击“查看对应原文”即可对照阅读。</p>
<div class="status">'''+escape(status)+'</div></header>'+''.join(items)+glossary_section+'<footer>'+escape(result['notice'])+'</footer></main></html>'
