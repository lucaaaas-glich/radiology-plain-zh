"""Low-memory local Apple Silicon inference with the quantized trained student."""
import argparse
import json
import time
from pathlib import Path
from .prompts import messages
from .fidelity import numeric_review, print_numeric_warnings


def main():
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    p = argparse.ArgumentParser(description="本地中文影像报告解释 · Apple Silicon")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--text")
    source.add_argument("--input", type=Path)
    p.add_argument("--model", default="artifacts/mlx-4bit")
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    if not Path(a.model).is_dir():
        p.error(f"本地模型尚未准备完成：{a.model}")
    model, tokenizer = load(a.model)
    once = a.text is not None or a.input is not None
    if not once:
        print("影像报告，说得明白。输入英文或中文报告，空行结束输入；输入 q 退出。\n研究原型，解释需要与原报告核对。")
    while True:
        if a.input:
            report = a.input.read_text(encoding="utf-8").strip()
        elif a.text is not None:
            report = a.text.strip()
        else:
            print("\n粘贴报告：")
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    return
                if line.strip().lower() == "q" and not lines:
                    return
                if not line.strip():
                    break
                lines.append(line)
            report = "\n".join(lines).strip()
        if not report:
            if once:
                p.error("输入不能为空")
            continue
        prompt = tokenizer.apply_chat_template(messages(report), tokenize=False, add_generation_prompt=True)
        if len(tokenizer.encode(prompt)) > 2048:
            print("报告超过本原型的输入长度限制；请使用较短的完整报告，不会自动截掉后文。")
            if once:
                return
            continue
        start, parts, final = time.perf_counter(), [], None
        for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=a.max_tokens,
                                         sampler=make_sampler(temp=0)):
            parts.append(response.text)
            final = response
            if not a.json:
                print(response.text, end="", flush=True)
        text = "".join(parts).strip()
        review = numeric_review(report, text)
        meta = {"seconds": time.perf_counter()-start, "model": a.model,
                "generation_tokens": getattr(final, "generation_tokens", None),
                "finish_reason": getattr(final, "finish_reason", None)}
        if a.json:
            print(json.dumps({"explanation_zh": text, "generation": meta, "numeric_review": review}, ensure_ascii=False, indent=2))
        else:
            print()
            print_numeric_warnings(review)
            if getattr(final, "generation_tokens", 0) >= a.max_tokens:
                print("[输出达到长度上限，可能不完整。]")
        if once:
            return


if __name__ == "__main__":
    main()
