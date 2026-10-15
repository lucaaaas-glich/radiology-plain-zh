"""Shared local inference backend for v2. No remote inference or silent truncation."""
import time
from pathlib import Path


class MLXBackend:
    def __init__(self, model_path, max_input_tokens=4096):
        if not Path(model_path).is_dir():
            raise ValueError(f"找不到本地模型目录：{model_path}")
        from mlx_lm import load
        start = time.perf_counter()
        self.model, self.tokenizer = load(model_path)
        self.load_seconds = time.perf_counter() - start
        self.model_path = str(model_path)
        self.max_input_tokens = max_input_tokens

    def __call__(self, messages, max_tokens=384):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        count = len(self.tokenizer.encode(prompt))
        if count > self.max_input_tokens:
            raise ValueError(f"输入需要{count}个token，超过本原型{self.max_input_tokens}的限制；未截断输入。")
        parts, last, start = [], None, time.perf_counter()
        for response in stream_generate(self.model, self.tokenizer, prompt=prompt,
                                        max_tokens=max_tokens, sampler=make_sampler(temp=0)):
            parts.append(response.text)
            last = response
        return {"text": "".join(parts).strip(), "seconds": time.perf_counter() - start,
                "input_tokens": count, "generation_tokens": getattr(last, "generation_tokens", 0),
                "finish_reason": getattr(last, "finish_reason", None)}
