"""Post-hoc application checks; these flags are not model accuracy scores."""
from pathlib import Path
from radplain.common import read_jsonl, write_json
from radplain.fidelity import numeric_review

root = Path("results/pilot_v1")
result = {"scope": "Post-pilot number/unit overlap flags; may miss errors or flag valid paraphrases. No clinical-accuracy interpretation.", "sets": {}}
for name, path in [("challenge-baseline", root/"challenge-baseline/predictions.jsonl"),
                   ("challenge-lora", root/"challenge-lora/predictions.jsonl"),
                   ("mac-4bit", Path("results/mac_4bit_v1/predictions.jsonl"))]:
    if path.exists():
        rows = read_jsonl(path)
        result["sets"][name] = {"total": len(rows), "flags": []}
        for row in rows:
            review = numeric_review(row["report"], row["prediction"])
            if review["warnings"]:
                result["sets"][name]["flags"].append({"id": row["id"], **review})
write_json(root/"posthoc_numeric_flags.json", result)
