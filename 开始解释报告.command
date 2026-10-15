#!/bin/zsh
cd "$(dirname "$0")"
if [[ ! -x .venv-mac/bin/python ]]; then
  print "本地运行环境尚未准备好，请先查看 README。"
  read -k 1
  exit 1
fi
.venv-mac/bin/python -m radplain.predict_mlx
