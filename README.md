# 影像报告，说得明白 · Radiology Plain Chinese

把专业的放射科报告改写成患者更容易理解的简体中文。一个独立的学习与研究项目，源于 COMP3710 的放射科报告通俗化选题，但不作为第二份课程作业提交。

**V3 的 20 份真实长报告主测试已完成，整体没有超过旧版；固定 10 份去清单对照仍在运行。** 旧版与新版各有 4 份发现明确错误；讲全且未发现事实或精度问题的报告从 5 份降到 1 份，累计生成耗时从 45.0 分钟增到 122.1 分钟。使用同一学校 27B 模型，没有重新训练权重。以上是项目 AI 对照原文的审阅，非医学准确率。V1/V2 及默认本机入口保留，不能将主测试已完成误认为全部 50 次输出已齐备。

见[事实清单怎么工作](docs/v3/事实清单怎么工作.md)、[实际核对过程实例](docs/v3/核对过程实例.md)、[复现与使用](docs/v3/复现与使用.md)、[预测前测试规模修订](docs/v3/测试规模修订.md)和[续跑与收尾安排](docs/v3/续跑与收尾安排.md)。V3 的 [PR 3](https://github.com/lucaaaas-glich/radiology-plain-zh/pull/3) 当前为草稿，最终对比和中文结果说明完成后再更新。

**V2研究交付与五方案最终评估已完成。** 本机4B入口可输出逐句中文、对应原文、词义和未完成提示；学校27B结果已取回并补齐复核。第一版及原始结果保留。

48条预留合成短报告的完整忠实解释：V1为23份，本机V2为43份，学校27B为45份。24份预留真实报告分别为3、4、10份；27B仍有5份发现错误、7份无明确错误但不完整、2份精度顾虑。27B规则重检虽然将有错报告降至4份，却把完整忠实解释降至8份。这是研究原型，不能称为可靠的患者报告解释器；所有质量数字均为项目AI复核，非临床准确率。

见[最终五方案实测与讲解](docs/v2/最终五方案实测与讲解.md)、[最终结果复现](docs/v2/最终评估复现.md)、[本机详细报告](docs/v2/本机实测与讲解.md)和[方法与使用](docs/v2/方法与使用.md)。[冻结前候选选择](docs/v2/候选选择.md)及[预先评估计划](docs/v2/评估计划.md)保持原样。真实原文留在忽略目录，GitHub保存逐例问题摘要与散列。14:00的[窗口结束说明](docs/v2/窗口结束说明.txt)是访问恢复前的历史快照，后续状态以最终报告为准。

双击`开始解释报告_V2.command`使用已经固定的本机4B方案；27B的A100配置峰值CUDA分配17.18GiB，未改成本机默认。额外NLLB翻译探针已因医学术语错误被拒绝。下文保留V1训练与使用记录。

**V1完成情况：A100训练、基线对比、Mac量化与真实本地运行验证均已完成。** 文字指标提高，但仍观察到数值和确定性错误。完整证据见 [实测结果与讲解](docs/实测结果与讲解.md)。

在249条具有可用合成中文参考的测试报告上，ROUGE-L从0.409提高到0.553，chrF从26.89提高到37.33。实际使用4,066条训练样本，A100训练9.4分钟。分数表示与教师文本的接近程度，**不表示医疗准确率**。例如微调模型曾把5毫米增大到8毫米说成“增大1毫米”，也曾改错随访间隔。

## 它做什么

输入英文或中文影像报告，输出通俗中文，尽量保留否定、左右侧、数字、变化和不确定性。
例如，“可能是感染，不能排除肿瘤”不能改成“确诊肿瘤”，也不能删去“不能排除”。
项目限定为文本解释：不读取影像、不新增诊断、不推荐原文没有的治疗。

## 研究设计

1. 从 BioLaySumm LaymanRRG 官方英文报告与通俗参考中选择固定子集。
2. 先按标准化后的报告文本去重，排除训练与官方 validation 的文本交叉。
3. 用冻结的 Qwen2.5-7B-Instruct 生成中文通俗参考。它们是**合成标签，未经医生验证**。
4. 对 Qwen2.5-1.5B-Instruct 做 LoRA 微调。只计算回答部分的训练损失。
5. 以开发集回答损失选择 checkpoint，在固定测试集上比较未微调模型与 LoRA 模型。
6. 额外检查预先编写的中英文挑战样例，观察否定、不确定性、左右侧和数字是否保留。

正式训练计划为 4,096 条英文报告到中文解释的样本，开发集和最终测试集各 256 条。
后两者来自官方 validation 的不相交子集；官方 test 不使用。过长或未完整生成的训练样本会显式记录并排除。
训练主要覆盖**英文输入到中文输出**。中文输入依赖基础模型的多语能力，只在小型合成挑战集上探索，不能声称已经验证中文临床泛化。

选择 Qwen 是因为此项目优先探索中文能力，不受课程 T5/FLAN-T5/GPT-2 模型清单约束。

## 使用

### 本机 Apple Silicon Mac

本机独立的 `.venv-mac` 环境和4位模型已经准备好。双击项目里的 **`开始解释报告.command`**，粘贴英文或中文报告，再输入一个空行，即可得到中文解释。输入 `q` 退出。

实测模型权重868.6MB，20条短报告全部生成结束，单条生成时间中位数0.62秒；MLX记录的峰值分配内存1.12GiB。这不含每次启动的模型加载，也不代表长报告速度或整个进程内存。[本地测量记录](results/mac_4bit_v1/summary.json)

也可以从项目根目录运行：

```bash
.venv-mac/bin/python -m radplain.predict_mlx --input examples/中文报告.txt
.venv-mac/bin/python -m radplain.predict_mlx --input examples/英文报告.txt --json
```

本地使用 `artifacts/mlx-4bit`。模型准备完成后，日常推理不需要学校GPU，输入文本不会由这些推理脚本发送到外部服务。首次运行会加载模型。模型尚未导出时程序会明确提示缺少模型，不会把其他模型冒充成训练结果。

### CUDA / 通用 PyTorch

在本项目目录建立独立环境，避免改动已有作业环境：

```bash
python -m venv .venv
source .venv/bin/activate
# 根据设备安装适合的 PyTorch，然后安装：
pip install -r requirements.txt
python -m radplain.predict --text 'No pleural effusion or pneumothorax.'
```

默认读取本地 `artifacts/base-model` 和 `artifacts/pilot_v1/best`。
也可传入 `--model Qwen/Qwen2.5-1.5B-Instruct` 下载基础模型，使用 `--baseline` 不加载 adapter。
模型权重、真实数据与完整预测不提交到 Git。学校项目目录保留基础模型及适配器；Mac使用单独导出的MLX版本。新的独立克隆需要按下方实验步骤生成权重，Git仓库本身不含可直接推理的模型文件。

```bash
python -m radplain.predict --input report.txt --device cuda --json
python -m radplain.predict --text '左侧少量胸腔积液，右侧未见积液。' --device cuda
```

## Rangpur A100 复现

只在独立项目目录操作。使用学校已存在的 PyTorch 环境作为只读基础，新建本项目 venv：

```bash
~/miniconda3/envs/torch/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
mkdir -p logs
/opt/slurm/bin/sbatch scripts/run_a100.sh
```

脚本在 Slurm 分配的 `/scratch/JOB_ID` 缓存模型，正式产物写入本项目 `artifacts/` 和 `results/`。
训练前会运行数据/损失掩码/中文指标单元测试，并在真实 A100 上检查有限非零梯度。
不要重复提交同一实验；训练与评估目录已存在时程序会拒绝覆盖。

复现已保存的实验时，在**新的独立克隆目录**中指定模型锁文件和数据 revision，避免上游 `main` 的更新改变数据或权重：

```bash
export RADZH_MODEL_LOCK="$PWD/results/pilot_v1/models.lock.json"
export RADZH_DATA_REVISION=6783ab38f9de28c4372799346f51d3927ae7eb9a
/opt/slurm/bin/sbatch scripts/run_a100.sh
```

锁文件只使用模型名称和 revision，其中旧机器的缓存路径不参与下载。固定版本、划分和随机种子帮助复现实验，但不同硬件和软件下不承诺逐位相同的训练结果。

## 怎样理解分数

- jieba 分词后的 ROUGE-1/2/L 和 chrF 衡量与合成参考的文字接近程度。
- 中文比例、输出是否到达结束标记、每秒生成 token 数用于检查语言与运行表现。
- ROUGE 默认分词会丢弃中文，因此本项目显式提供中文 tokenizer。
- 这些指标**不等于医疗准确率、可读性验证或临床安全性**。
- 教师和学生来自同一模型家族，模仿教师的分数可能高估真正的事实改善。
- 文本去重不等于患者独立；缺少患者 ID 时不能证明患者级无泄漏。
- 20 个挑战样例由 AI 编写，用于观察行为，不能替代独立临床评估。

详细实验边界见 [实验设计与局限](docs/实验设计与局限.md)，概念讲解见 [学习路线](docs/学习路线.md)。

本地命令行另有数字与单位核对提示。这是在发现试验错误后加入的独立规则，不修改生成文本，也不影响上述原始指标；没有提示不等于事实正确。完整的20例合成挑战对比见 [逐例分析](results/pilot_v1/challenge_review.json)。

## 文件说明

| 文件 | 作用 |
|---|---|
| `configs/pilot.json` | 预先确定的实验规模和超参数 |
| `radplain/data.py` | 数据下载、去重、固定划分与来源记录 |
| `radplain/teacher.py` | 可续跑的中文合成标签生成 |
| `radplain/train.py` | 回答损失掩码、LoRA 训练、开发集选模 |
| `radplain/evaluate.py` | 相同设置下生成与评估 |
| `radplain/predict.py` | 中文命令行推理 |
| `examples/challenge.jsonl` | 预先编写的合成挑战集 |
| `scripts/run_a100.sh` | 学校集群的一次完整实验 |

## 来源与许可

- [BioLaySumm LaymanRRG 数据集](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track)：只通过官方来源取得，记录 revision；本仓库不重新分发其训练或评估数据。
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 和 [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)：基础模型使用其各自 Apache-2.0 许可。
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model) 与 [PEFT LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)：使用现成模型与 adapter 实现，本项目没有从头编写 Qwen 网络或预训练。

## AI usage disclosure

Codex assisted with research design, code, tests, experiment operations, analysis and Chinese documentation.
Qwen generates synthetic Chinese training targets. Automated checks and AI-assisted case review do not constitute clinician review.
Every reported measurement must be linked to saved real-run evidence. No training result is invented.
