"""Make a compact, auditable comparison from actual measured outputs."""
import json
from pathlib import Path
from radplain.common import write_json

root = Path("results/pilot_v1")
baseline = json.loads((root / "baseline/metrics.json").read_text())
lora = json.loads((root / "lora/metrics.json").read_text())
training = json.loads(Path("artifacts/pilot_v1/training.json").read_text())
manifest = json.loads(Path("data/pilot_v1/manifest.json").read_text())
teacher = json.loads(Path("data/pilot_v1/teacher_summary.json").read_text())
selection = json.loads(Path("artifacts/pilot_v1/selection.json").read_text())
write_json(root / "comparison.json", {"baseline": baseline, "lora": lora, "training": training,
                                      "data": manifest, "teacher": teacher, "selection": selection})
lines = ["# 实测结果 / Measured results", "", "中文参考由模型生成；下列文本相似度不代表医疗准确率。", "",
         "| 指标 | 未微调基线 | LoRA |", "|---|---:|---:|"]
for key in ["rouge1", "rouge2", "rougeL", "chrf", "chinese_output_fraction", "generation_completed_fraction", "output_tokens_per_second"]:
    lines.append(f"| {key} | {baseline[key]:.4f} | {lora[key]:.4f} |")
lines.extend(["", f"训练耗时：{training['elapsed_seconds']/60:.1f} 分钟。",
              f"GPU：{training['gpu']}。训练峰值 allocated：{training['peak_allocated_bytes']/1024**3:.2f} GiB。",
              f"训练样本：{training['filters']['train']['used']}。最终评估样本：{baseline['total_rows']}。",
              f"依据验证集损失选取第 {selection['epoch']} 轮模型。",
              "", "病例错误分析必须另外阅读实际生成文本；不能只凭这些数值宣称安全或有效。"])
(root / "SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
print("Comparison written", flush=True)
