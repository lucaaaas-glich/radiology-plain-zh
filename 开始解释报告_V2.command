#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if ! test -x .venv-mac/bin/python; then
  printf '找不到本项目的 Python 环境，请先按 README 安装。\n'
  exit 1
fi
if ! test -f configs/v2_local.json; then
  printf 'V2 模型还在比较中，完成选型后才能使用此入口。\n'
  exit 1
fi
exec .venv-mac/bin/python -m radplain.predict_v2 --html-dir data/v2/explanations --open-html
