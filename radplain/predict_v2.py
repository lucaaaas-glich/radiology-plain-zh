"""Local v2 CLI. Buffer and check candidates before showing an explanation."""
import argparse
import json
from pathlib import Path
from .explain_v2 import explain, render_text
from .mlx_backend import MLXBackend


def public_result(result):
    """Raw failed candidates belong to experiment logs, not patient output."""
    return dict(result, segments=[{k:v for k,v in r.items() if k != 'attempts'} for r in result['segments']])


def main():
    p=argparse.ArgumentParser(description='中文影像报告解释 v2 · 原文对应与术语说明')
    s=p.add_mutually_exclusive_group()
    s.add_argument('--text')
    s.add_argument('--input',type=Path)
    p.add_argument('--model',help='覆盖已选定的本地模型；不自动下载')
    p.add_argument('--max-tokens',type=int,default=384)
    p.add_argument('--json',action='store_true')
    a=p.parse_args()
    if a.max_tokens <= 0:
        p.error('--max-tokens 必须大于0')
    config=Path('configs/v2_local.json')
    if a.model:
        model=a.model
    elif config.exists():
        model=json.loads(config.read_text())['model']
    else:
        p.error('v2模型选择尚未完成，请显式指定 --model')
    try:
        backend=MLXBackend(model)
    except ValueError as e:
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
            result=explain(report,backend,max_tokens=a.max_tokens)
        except ValueError as e:
            if once:
                p.error(str(e))
            print(str(e))
            continue
        result['model']=model
        result['model_load_seconds']=backend.load_seconds
        if a.json:
            print(json.dumps(public_result(result),ensure_ascii=False,indent=2))
        else:
            print(render_text(result))
        if once:
            return


if __name__=='__main__':
    main()
