# 实测结果 / Measured results

中文参考由模型生成；下列文本相似度不代表医疗准确率。

| 指标 | 未微调基线 | LoRA |
|---|---:|---:|
| rouge1 | 0.4530 | 0.5996 |
| rouge2 | 0.1988 | 0.3467 |
| rougeL | 0.4090 | 0.5527 |
| chrf | 26.8872 | 37.3349 |
| chinese_output_fraction | 0.9766 | 1.0000 |
| generation_completed_fraction | 1.0000 | 1.0000 |
| output_tokens_per_second | 144.5688 | 96.8405 |

训练耗时：9.4 分钟。
GPU：NVIDIA A100-PCIE-40GB。训练峰值 allocated：10.10 GiB。
训练样本：4066。最终评估样本：256。
依据验证集损失选取第 2 轮模型。

病例错误分析必须另外阅读实际生成文本；不能只凭这些数值宣称安全或有效。
