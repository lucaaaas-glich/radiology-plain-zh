"""Explicit model/backend CLI; V2 remains the existing local default."""
import argparse
import json
from pathlib import Path
import sys
from .explain_v3 import explain
from .render_v3 import render_html, render_text, public_result


def main():
    p=argparse.ArgumentParser(description='V3 原文事实清单与中文报告解释（研究原型）')
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True,help='已存在的本地权重；不自动下载')
    p.add_argument('--backend',choices=['mlx','cuda-nf4'],required=True)
    p.add_argument('--html',type=Path)
    p.add_argument('--json-out',type=Path)
    a=p.parse_args()
    if not a.model.is_dir():p.error('模型目录不存在')
    for path in (a.html,a.json_out):
        if path and path.exists():p.error('输出已存在，拒绝覆盖：'+str(path))
    report=a.input.read_text(encoding='utf-8')
    if a.backend=='mlx':
        from .mlx_backend import MLXBackend
        print('提示：本机小模型与学校27B评估不是同一模型，不能直接套用27B结果。',file=sys.stderr)
        backend=MLXBackend(str(a.model),max_input_tokens=8192)
    else:
        from .hf_backend_v2 import HFBackend
        backend=HFBackend(str(a.model),max_input_tokens=8192,load_in_4bit=True)
    result=explain(report,backend,progress=lambda done,total:print(f'已生成中文片段：{done}/{total}',file=sys.stderr,flush=True))
    result.update(model_path=str(a.model),backend=a.backend,model_load_seconds=backend.load_seconds)
    if a.html:
        a.html.parent.mkdir(parents=True,exist_ok=True)
        with a.html.open('x',encoding='utf-8') as f:f.write(render_html(result))
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open('x',encoding='utf-8') as f:json.dump(public_result(result),f,ensure_ascii=False,indent=2)
    print(render_text(result))


if __name__=='__main__':main()
