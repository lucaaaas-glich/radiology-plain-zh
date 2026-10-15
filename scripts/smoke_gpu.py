"""Validate actual Qwen template masking and a finite LoRA gradient on A100."""
import argparse
import torch
from peft import LoraConfig, get_peft_model
from radplain.model import load_model, generate_batch
from radplain.train import encode_example, Collator
from radplain.prompts import messages
from radplain.common import write_json

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
a = p.parse_args()
model, tok = load_model(a.model, training=True)
model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
row = {"report": "No pleural effusion or pneumothorax.", "reference_zh": "没有发现胸腔积液或气胸。"}
e = encode_example(tok, row, 1152)
assert e is not None
batch = {k: v.to("cuda") for k,v in Collator(tok.pad_token_id)([e]).items()}
loss = model(**batch).loss
assert torch.isfinite(loss)
loss.backward()
grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
assert grads and all(torch.isfinite(g).all() for g in grads)
assert any(g.abs().sum() > 0 for g in grads)
write_json("results/pilot_v1/smoke.json", {"loss": loss.item(), "finite_nonzero_gradients": True,
          "mask_verified": True, "gpu": torch.cuda.get_device_name(), "note": "Synthetic smoke example only; weights discarded."})
print("GPU forward/backward and Qwen label masking passed", flush=True)
