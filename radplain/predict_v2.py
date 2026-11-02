"""Local v2 CLI. Buffer and check candidates before showing an explanation."""
import argparse
import json
import sys
import uuid
import webbrowser
from pathlib import Path
from .explain_v2 import explain, render_text
from .render_v2 import render_html


def public_result(result):
    """Raw failed candidates belong to experiment logs, not patient output."""
    return dict(result, segments=[{k:v for k,v in r.items() if k != 'attempts'} for r in result['segments']])


def main():
    p=argparse.ArgumentParser(description='中文影像报告解释 v2 · 原文对应与术语说明')
    s=p.add_mutually_exclusive_group()
    s.add_argument('--text')
    s.add_argument('--input',type=Path)
    p.add_argument('--model',help='覆盖已选定的本地模型；不自动下载')
    p.add_argument('--backend',choices=['mlx','cuda','cuda-nf4'],default='mlx')
    p.add_argument('--mode',choices=['evidence','evidence-critic'],help='覆盖配置里的解释方法')
    p.add_argument('--max-tokens',type=int,default=384)
    p.add_argument('--json',action='store_true')
    p.add_argument('--html',type=Path,help='将一次解释保存为独立网页；需同时指定 --text 或 --input')
    p.add_argument('--html-dir',type=Path,help='每次解释保存到此目录，生成独立网页')
    p.add_argument('--open-html',action='store_true',help='解释后用默认浏览器打开生成的本地网页')
    a=p.parse_args()
    if a.max_tokens <= 0:
        p.error('--max-tokens 必须大于0')
    if a.html and a.text is None and a.input is None:
        p.error('--html 需同时指定 --text 或 --input')
    if a.html and a.html.exists():
        p.error('--html 目标已存在，拒绝覆盖')
    if a.html and a.html_dir:
        p.error('--html 和 --html-dir 不能同时使用')
    if a.open_html and not (a.html or a.html_dir):
        p.error('--open-html 需要 --html 或 --html-dir')
    config=Path('configs/v2_local.json' if a.backend=='mlx' else 'configs/v2_gpu.json')
    settings=json.loads(config.read_text()) if config.exists() else {}
    if a.model:
        model=a.model
    elif 'model' in settings:
        model=settings['model']
    else:
        p.error('v2模型选择尚未完成，请显式指定 --model')
    mode=a.mode or settings.get('mode','evidence')
    if mode not in {'evidence','evidence-critic'}:
        p.error('配置里的 mode 无效')
    try:
        if not a.json:
            print('正在加载本地模型……', file=sys.stderr, flush=True)
        if a.backend=='mlx':
            from .mlx_backend import MLXBackend
            backend=MLXBackend(model)
        else:
            from .hf_backend_v2 import HFBackend
            backend=HFBackend(model,load_in_4bit=a.backend=='cuda-nf4')
    except (ValueError,RuntimeError) as e:
        p.error(str(e))
    once=a.text is not None or a.input is not None
    if not once:
        print('中文影像报告解释 v2\n粘贴完整报告，空行结束；q退出。解释会与原句一起显示。\n研究原型，内容需要对照原报告。')
    while True:
        if a.input:
            try:
                report=a.input.read_text(encoding='utf-8')
            except OSError as e:
                p.error(str(e))
        elif a.text is not None:
            report=a.text
        else:
            print('\n粘贴报告：')
            lines=[]
            while True:
                try:
                    line=input()
                except EOFError:
                    return
                if not lines and line.strip().lower()=='q':
                    return
                if not line.strip():
                    break
                lines.append(line)
            report='\n'.join(lines)
            if not report.strip():
                continue
        try:
            if not a.json:
                print('正在逐句解释和核对，请稍候……', file=sys.stderr, flush=True)
            def progress(done, total, status):
                label={'generated':'已生成中文，仍需对照原文',
                       'source_only':'本句未完成，保留原文',
                       'excluded_instruction':'已跳过非影像指令'}[status]
                print(f'进度 {done}/{total}：{label}', file=sys.stderr, flush=True)
            result=explain(report,backend,max_tokens=a.max_tokens,use_critic=mode=='evidence-critic',
                           progress=None if a.json else progress)
        except (ValueError, RuntimeError) as e:
            if once:
                p.error(str(e))
            print(str(e))
            continue
        result['model']=model
        result['backend']=a.backend
        result['mode']=mode
        result['model_load_seconds']=backend.load_seconds
        html_path=a.html
        if a.html_dir:
            html_path=a.html_dir/('解释-'+uuid.uuid4().hex[:12]+'.html')
        if html_path:
            html_path.parent.mkdir(parents=True,exist_ok=True)
            with html_path.open('x',encoding='utf-8') as f:
                f.write(render_html(result))
        if a.json:
            print(json.dumps(public_result(result),ensure_ascii=False,indent=2))
        else:
            print(render_text(result))
            if html_path:
                print('已保存：'+str(html_path.resolve()))
        if a.open_html:
            webbrowser.open(html_path.resolve().as_uri())
        if once:
            return


if __name__=='__main__':
    main()
