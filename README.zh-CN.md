[English](./README.md) | [简体中文](./README.zh-CN.md)

# NVIDIA Nemotron Model Reasoning Challenge

这是我参加 Kaggle `NVIDIA Nemotron Model Reasoning Challenge` 的开源工程记录。

这个仓库不只是放 notebook，而是把整段比赛经历工程化下来：

- 复现公开 `SFT baseline`
- 生成和清洗 `CoT` 数据
- 搭建本地 dry-run / 配置驱动训练脚本
- 记录 Kaggle 离线环境下的踩坑过程
- 迭代 Stage 2c notebook，并验证数据与训练逻辑


## 项目目标

目标是微调 `Nemotron-3-Nano-30B-A3B-BF16`，提升它在规则推理题上的准确率。

比赛任务包含 4 类题型：

- bit manipulation
- text transformation / encryption
- number system conversion
- unit conversion

评测方式是从模型输出中提取最终的 `\boxed{...}` 答案，并进行精确匹配或数值容差比较。

## 当前进度

| Stage | 说明 | 结果 |
|---|---|---|
| Stage 1 | 复现公开 SFT baseline | `0.67` |
| Stage 2a | 仅用 435 条 CoT 从 base model 新建 LoRA | `0.59` |
| Stage 2b | 直接答案 + CoT 混合，从 base model 新建 LoRA | `0.62` |
| Stage 2c 首版 | baseline 继续训练，但实现存在问题 | `0.58` |
| Stage 2c 修正版 | 已完成本地重构与验证，待 Kaggle 重跑 | `pending` |

更完整的实验记录见 [record.md](./record.md)。

## 仓库里有什么

### Kaggle notebooks

- `notebooks/nvidia-nemotron-sfttrainer-training.ipynb`
  - Stage 1 baseline notebook
- `notebooks/nvdia-nemotron-cot-sft.ipynb`
  - Stage 2c 修正版 notebook

### 本地工程化脚本

- `train.py`
  - 配置驱动训练入口，支持 `--dry-run`
- `data_utils.py`
  - 训练数据读取、格式化、ChatML 转换
- `generate_cot.py`
  - 调用 OpenAI-compatible API 生成 CoT 数据
- `eval_local.py`
  - 本地提取 `\boxed{}` 并计算准确率
- `package_submission.py`
  - 将 adapter 打包成 `submission.zip`
- `runtime_patches.py`
  - Nemotron / Mamba 相关兼容补丁

### 配置与脚本

- `configs/sft_baseline.yaml`
- `configs/sft_cot.yaml`
- `scripts/setup_env.ps1`
- `scripts/run_train.ps1`
- `scripts/run_cot_generation.ps1`
- `scripts/validate_stage2c_notebook.py`

### 经验复盘

- `record.md`
  - 从 Stage 1 到 Stage 2c 的阶段记录、错误复盘、下一步计划
- `docs/notebook_migration.md`
  - 从 Kaggle notebook 拆到本地项目结构的迁移说明
- `预案.md`
- `预算.md`

## 仓库结构

```text
.
├─ configs/
├─ data/
├─ docs/
├─ notebooks/
├─ output/
├─ prompts/
├─ scripts/
├─ train.py
├─ data_utils.py
├─ generate_cot.py
├─ eval_local.py
├─ package_submission.py
├─ runtime_patches.py
├─ requirements.txt
├─ Dockerfile
└─ record.md
```

## 关键结论

这次比赛里，我目前最重要的几个结论是：

1. `CoT` 不是万能药，小样本 CoT 从头训练很容易低于 baseline。
2. 对数据量受限的场景，优先做“在 baseline adapter 上继续训练”，不要轻易重新从 base model 起一个新 LoRA。
3. `packing=True` 在 Kaggle 这个 Nemotron 离线环境里风险很大，因为没有可靠的 Flash Attention 支撑，可能造成样本间注意力污染。
4. 对这类规则执行题，训练格式的一致性非常重要，尤其是最终答案的 `\boxed{}` 位置和数量。
5. notebook 跑通不等于方案正确，Stage 2c 的低分有相当一部分来自实现问题，而不是策略本身。

## Stage 2c 修正版做了什么

修正版 `notebooks/nvdia-nemotron-cot-sft.ipynb` 主要修了这几件事：

- 直接加载 baseline adapter 继续训练
  - 使用 `PeftModel.from_pretrained(..., is_trainable=True)`
  - 不再 `merge_and_unload()`
  - 不再重新新建 LoRA
- 恢复 direct-answer 样本为 Stage 1 baseline 的原始格式
  - assistant 只输出 `\boxed{answer}`
- 先清洗 CoT 数据再训练
  - 修复 `\ ` 空格
  - 去掉 `reasoning` 尾部已有的 `\boxed{...}`
  - 保证 assistant 侧只保留一个最终 boxed 答案
- 将 CoT 过采样从 `4x` 降为 `2x`
  - 当前修正版使用 `5000` 条 direct-answer + `435 x 2 = 870` 条 CoT
- 关闭 `packing`
- 将 `max_seq_len` 提升到 `1536`
- 将继续训练学习率固定为 `5e-5`

本地已经用 `scripts/validate_stage2c_notebook.py` 做过一轮数据与逻辑验证，当前验证结果是：

- `435` 条 CoT 全部通过清洗保留
- `56` 条 LaTeX 转义空格样本被归一化
- 最终混合训练样本数为 `5870`
- 预估 optimizer steps 约 `1467`

## 本地快速开始

### 1. 环境准备

Windows PowerShell:

```powershell
.\scripts\setup_env.ps1
Copy-Item .env.example .env
```

然后手动填写 `.env`：

- `COT_API_KEY`
- `COT_BASE_URL`
- `COT_MODEL`

### 2. 数据准备

将你自己的比赛数据放到 `data/` 目录下：

- `data/train.csv`
- `data/test.csv`
- `data/train_cot.jsonl`

说明：

- 我不建议在开源仓库里重新分发比赛原始数据
- 如果你 fork 这个仓库，请自行从 Kaggle 获取数据

### 3. 本地 dry run

```powershell
.\scripts\run_train.ps1 -Config configs/sft_baseline.yaml -DryRun -MaxSamples 8
```

### 4. 验证 Stage 2c 数据逻辑

```powershell
python scripts\validate_stage2c_notebook.py --write-cleaned data\train_cot_clean.jsonl
```

### 5. 生成 CoT

```powershell
.\scripts\run_cot_generation.ps1
```

## Kaggle 端运行说明

这个项目的真正大模型训练主要还是在 Kaggle 比赛 notebook 上完成，原因有两个：

- 比赛提供 `RTX Pro 6000 Blackwell 96GB`
- Nemotron 30B 的完整训练和提交通路都更贴近 Kaggle 官方环境

Kaggle 端需要特别注意：

- 比赛 notebook 强制断网
- `trl` / `peft` 等依赖需要用离线包安装
- `mamba_ssm` / `causal_conv1d` 可能不存在，需要 mock / patch
- 不要盲目开启 `packing`

## 复现边界

这个仓库包含的是“工程代码 + notebook + 复盘”，不是一键满血复现包。

你仍然需要自己准备：

- Kaggle 比赛数据
- Kaggle 比赛环境
- OpenAI-compatible API key
- 足够的 GPU 资源

此外，这里不包含：

- Nemotron 基座模型权重
- 任何私密 API key
- 保证稳定涨分的最终答案

## 这个项目带给我的收获

如果只用一句话总结，这个项目教会我的不是“怎么把一个 notebook 跑通”，而是：

**怎么把一个分散的比赛实验，逐步整理成可以验证、可以复盘、可以分享的工程。**

我把很多失败案例也保留了下来，因为这些失败比“最后某个分数”更有复用价值：

- 为什么纯 CoT 会掉分
- 为什么 packing 会把训练搞坏
- 为什么 merge 后再套新 LoRA 不是继续训练
- 为什么输出格式漂移会影响最终成绩
- 为什么小样本 CoT 一定要先做数据清洗

## 相关文件

- [record.md](./record.md)
- [docs/notebook_migration.md](./docs/notebook_migration.md)
- [notebooks/nvidia-nemotron-sfttrainer-training.ipynb](./notebooks/nvidia-nemotron-sfttrainer-training.ipynb)
- [notebooks/nvdia-nemotron-cot-sft.ipynb](./notebooks/nvdia-nemotron-cot-sft.ipynb)

## 致谢

- Kaggle 比赛官方
- NVIDIA Nemotron 团队
- Dennis Fong 提供的公开 baseline notebook

如果这份仓库对你有帮助，欢迎 fork、提 issue，或者直接基于它继续做自己的比赛工程化版本。
