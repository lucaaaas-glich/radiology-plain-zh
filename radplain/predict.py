"""Command-line Chinese explanation of English or Chinese radiology text."""
import argparse
import json
from pathlib import Path
from .model import generate_batch, load_model
from .prompts import messages


def main():
    p = argparse.ArgumentParser(description="放射科报告通俗中文改写（研究原型）")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="原始报告文本")
    source.add_argument("--input", type=Path, help="UTF-8 文本文件")
    p.add_argument("--model", default="artifacts/base-model")
    p.add_argument("--adapter", default="artifacts/pilot_v1/best")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"])
    p.add_argument("--json", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=384)
    a = p.parse_args()
    report = a.input.read_text(encoding="utf-8").strip() if a.input else a.text.strip()
    if not report:
        p.error("输入不能为空")
    model, tok = load_model(a.model, None if a.baseline else a.adapter, device=a.device)
    text, meta = generate_batch(model, tok, [messages(report)], max_input_tokens=2048,
                                max_new_tokens=a.max_new_tokens)
    if a.json:
        print(json.dumps({"explanation_zh": text[0], "generation": meta[0],
                          "notice": "研究原型；请核对原报告。"}, ensure_ascii=False, indent=2))
    else:
        print(text[0])
        if not meta[0]["completed"]:
            print("\n[提示：输出达到长度上限，可能不完整。]")


if __name__ == "__main__":
    main()
