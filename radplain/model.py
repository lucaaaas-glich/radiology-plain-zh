"""Model loading and generation with explicit length and completion reporting."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name, adapter=None, training=False, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="right" if training else "left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=training)
    model.to(device)
    model.train(training)
    model.config.use_cache = not training
    return model, tokenizer


def generate_batch(model, tokenizer, conversations, max_input_tokens=1536, max_new_tokens=384):
    prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in conversations]
    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    if max(lengths, default=0) > max_input_tokens:
        raise ValueError(f"Input exceeds {max_input_tokens} tokens; refusing silent truncation: {max(lengths)}")
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tail = output[:, inputs.input_ids.shape[1]:]
    text = tokenizer.batch_decode(tail, skip_special_tokens=True)
    meta = []
    for ids, length in zip(tail.tolist(), lengths):
        eos_at = next((i for i, x in enumerate(ids) if x == tokenizer.eos_token_id), None)
        count = eos_at + 1 if eos_at is not None else len(ids)
        meta.append({"input_tokens": length, "output_tokens": count,
                     "completed": eos_at is not None, "batch_seconds": elapsed})
    return [x.strip() for x in text], meta
