"""Paired lexical uncertainty and targeted case review, never clinical accuracy."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from rouge_score.rouge_scorer import RougeScorer
from radplain.common import read_jsonl, write_json
from radplain.evaluate import ChineseTokenizer

p = argparse.ArgumentParser()
p.add_argument("--root", default="results/pilot_v1")
a = p.parse_args()
root = Path(a.root)
b = read_jsonl(root/"baseline/predictions.jsonl")
f = read_jsonl(root/"lora/predictions.jsonl")
assert [r["id"] for r in b] == [r["id"] for r in f], "Evaluation memberships/order differ"
assert len({r["id"] for r in b}) == len(b), "Duplicate test IDs"
assert all(x["report"] == y["report"] and x["reference_zh"] == y["reference_zh"] for x,y in zip(b,f)), "Evaluation sources/references differ"
scorer = RougeScorer(["rougeL"], tokenizer=ChineseTokenizer())
diffs = []
for x,y in zip(b,f):
    if x["teacher_accepted"]:
        base = scorer.score(x["reference_zh"],x["prediction"])["rougeL"].fmeasure
        adapted = scorer.score(y["reference_zh"],y["prediction"])["rougeL"].fmeasure
        diffs.append(adapted-base)
rng = np.random.default_rng(371026)
diffs = np.asarray(diffs)
means = rng.choice(diffs, size=(2000,len(diffs)), replace=True).mean(axis=1)
write_json(root/"paired_lexical_analysis.json", {"metric":"jieba_ROUGE_L_fmeasure", "n":len(diffs),
           "mean_lora_minus_baseline":float(diffs.mean()),
           "paired_bootstrap_95_percentile_interval":np.quantile(means,[.025,.975]).tolist(),
           "bootstrap_replicates":2000,"seed":371026,
           "limitations":"Conditional on sampled reports and synthetic references. Not clinical safety, not independent patient validation; one training seed."})
# Cases are selected by source content, not by which model looks better.
patterns = [("negation",r"no (?:pleural|evidence|acute|pneumothorax)"),
            ("uncertainty",r"cannot exclude|may represent|possible|suggestive"),
            ("number_or_followup",r"\b\d+(?:\.\d+)?\s*(?:mm|cm|months?|weeks?)\b"),
            ("change",r"improv|decreas|increas|unchanged|stable"),
            ("device",r"tube|catheter|pacemaker")]
cases, used = [], set()
for category,pattern in patterns:
    for x,y in zip(b,f):
        if x["id"] not in used and re.search(pattern,x["report"],re.I):
            cases.append({"review_focus":category,"id":x["id"],"report":x["report"],
                          "reference_zh_synthetic":x["reference_zh"],"baseline":x["prediction"],
                          "lora":y["prediction"],"review_status":"pending_AI_assisted_review_not_clinician_review"})
            used.add(x["id"])
            break
write_json(root/"case-review.json",cases)
print('Paired lexical analysis and source-selected review cases written')
