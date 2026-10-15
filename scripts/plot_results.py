"""Plot recorded measurements; run with matplotlib in the school environment."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("results/pilot_v1")
data = json.loads((root / "comparison.json").read_text())
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
colors = ("#336b86", "#d8793e")
keys = ["rouge1", "rouge2", "rougeL"]
x = np.arange(len(keys))
for offset, kind, label, color in [(-.18, "baseline", "Base 1.5B", colors[0]), (.18, "lora", "LoRA", colors[1])]:
    bars = a.bar(x + offset, [data[kind][k] for k in keys], .34, label=label, color=color)
    a.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
a.set(xticks=x, xticklabels=["ROUGE-1", "ROUGE-2", "ROUGE-L"], ylim=(0, 1), ylabel="Mean F1", title="Agreement with synthetic references (n=249)")
a.legend(frameon=False)
history = data["training"]["history"]
b.plot([v["epoch"] for v in history], [v["train_token_loss"] for v in history], "o-", color=colors[0], label="Train")
b.plot([v["epoch"] for v in history], [v["validation_token_loss"] for v in history], "o-", color=colors[1], label="Development")
b.set(xticks=[1, 2], xlim=(.8, 2.2), ylim=(0, 1.2), xlabel="Epoch", ylabel="Answer-token cross-entropy", title="Checkpoint selected using development loss")
b.legend(frameon=False)
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=.6)
    ax.set_axisbelow(True)
fig.suptitle("Measured learning signals, not clinical accuracy", fontsize=14, fontweight="bold")
fig.savefig(root / "figure_metrics.png", dpi=180)
plt.close(fig)
