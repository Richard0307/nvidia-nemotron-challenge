[English](./README.md) | [Chinese (Simplified)](./README.zh-CN.md)

# NVIDIA Nemotron Model Reasoning Challenge

This repository is my engineering log for the Kaggle **NVIDIA Nemotron Model Reasoning Challenge**.

Instead of keeping everything inside notebooks, I gradually turned the project into a more maintainable workflow:

- reproduce the public SFT baseline
- generate and clean CoT data
- build local dry-run and config-driven scripts
- document Kaggle offline-environment pitfalls
- iterate on a revised Stage 2c notebook and validate its data/training logic



## Overview

The goal is to fine-tune `Nemotron-3-Nano-30B-A3B-BF16` for rule-based reasoning tasks.

The competition includes four task families:

- bit manipulation
- text transformation / encryption
- number system conversion
- unit conversion

Evaluation extracts the final answer from `\boxed{...}` and compares it with the ground truth using exact match or numeric tolerance.

## Results So Far

| Stage | Description | Result |
|---|---|---|
| Stage 1 | Reproduced the public SFT baseline | `0.67` |
| Stage 2a | Trained a fresh LoRA from base model on 435 CoT samples only | `0.59` |
| Stage 2b | Mixed direct-answer + CoT data, still from base model | `0.62` |
| Stage 2c v1 | Continued from baseline idea, but implementation was flawed | `0.58` |
| Stage 2c revised | Rewritten locally and validated, waiting for a new Kaggle run | `pending` |

The full running log is in [record.md](./record.md).

## Repository Highlights

### Kaggle notebooks

- `notebooks/nvidia-nemotron-sfttrainer-training.ipynb`
  - Stage 1 baseline notebook
- `notebooks/nvdia-nemotron-cot-sft.ipynb`
  - revised Stage 2c notebook

### Local scripts

- `train.py`
  - config-driven training entrypoint with `--dry-run`
- `data_utils.py`
  - data loading, formatting, ChatML conversion
- `generate_cot.py`
  - CoT generation through an OpenAI-compatible API
- `eval_local.py`
  - local `\boxed{}` extraction and scoring
- `package_submission.py`
  - packages adapters into `submission.zip`
- `runtime_patches.py`
  - Nemotron / Mamba compatibility patches

### Configs and helpers

- `configs/sft_baseline.yaml`
- `configs/sft_cot.yaml`
- `scripts/setup_env.ps1`
- `scripts/run_train.ps1`
- `scripts/run_cot_generation.ps1`
- `scripts/validate_stage2c_notebook.py`

### Notes and write-ups

- `record.md`
  - stage-by-stage experiment log, failure analysis, next-step planning
- `docs/notebook_migration.md`
  - notes on migrating from Kaggle notebooks to a local project structure
- [Project plan notes](./%E9%A2%84%E6%A1%88.md)
- [Budget notes](./%E9%A2%84%E7%AE%97.md)

## Repository Layout

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

## Key Takeaways

The most important lessons from this project so far:

1. CoT is not a free win. Small-sample CoT training can easily underperform a solid baseline.
2. When data is limited, continuing from a baseline adapter is much safer than starting a fresh LoRA from the base model.
3. `packing=True` is risky in this Kaggle Nemotron setup because reliable Flash Attention support is not available.
4. Output-format consistency matters a lot for rule-execution tasks, especially the placement and duplication of `\boxed{}`.
5. A notebook that runs is not the same as a correct strategy. Part of the Stage 2c drop came from implementation mistakes, not just from the training idea itself.

## What Changed in the Revised Stage 2c

The revised `notebooks/nvdia-nemotron-cot-sft.ipynb` fixes the main problems from the first Stage 2c attempt:

- directly continues training from the baseline adapter
  - uses `PeftModel.from_pretrained(..., is_trainable=True)`
  - no `merge_and_unload()`
  - no fresh LoRA on top of a merged model
- restores direct-answer samples to the original Stage 1 target format
  - assistant output is only `\boxed{answer}`
- cleans CoT data before training
  - normalizes `\ ` spaces
  - removes any trailing `\boxed{...}` already present in `reasoning`
  - keeps only one final boxed answer on the assistant side
- reduces CoT oversampling from `4x` to `2x`
  - current revised mix: `5000` direct-answer samples + `435 x 2 = 870` CoT samples
- disables `packing`
- increases `max_seq_len` to `1536`
- lowers continued-training LR to `5e-5`

Local validation with `scripts/validate_stage2c_notebook.py` currently shows:

- all `435` CoT rows survive cleaning
- `56` rows with LaTeX escaped spaces are normalized
- final mixed training size is `5870`
- estimated optimizer steps are about `1467`

## Quick Start

### 1. Prepare the environment

Windows PowerShell:

```powershell
.\scripts\setup_env.ps1
Copy-Item .env.example .env
```

Then fill `.env` manually:

- `COT_API_KEY`
- `COT_BASE_URL`
- `COT_MODEL`

### 2. Prepare the data

Place your own competition files under `data/`:

- `data/train.csv`
- `data/test.csv`
- `data/train_cot.jsonl`

Notes:

- I do not recommend redistributing the competition data inside the open-source repo.
- If you fork this project, please obtain the data from Kaggle yourself.

### 3. Run a local dry run

```powershell
.\scripts\run_train.ps1 -Config configs/sft_baseline.yaml -DryRun -MaxSamples 8
```

### 4. Validate the revised Stage 2c data logic

```powershell
python scripts\validate_stage2c_notebook.py --write-cleaned data\train_cot_clean.jsonl
```

### 5. Generate CoT data

```powershell
.\scripts\run_cot_generation.ps1
```

## Running on Kaggle

Actual large-model training still happens on Kaggle for this project, mainly because:

- the competition provides `RTX Pro 6000 Blackwell 96GB`
- the final training and submission path is tightly coupled to the Kaggle competition environment

Kaggle-specific caveats:

- competition notebooks are offline
- `trl` / `peft` often need offline wheel installation
- `mamba_ssm` / `causal_conv1d` may be missing and need patching
- avoid enabling `packing` blindly

## Reproducibility Boundaries

This repo contains engineering code, notebooks, and failure analysis. It is not a one-click full reproduction bundle.

You still need your own:

- Kaggle competition data
- Kaggle competition environment
- OpenAI-compatible API key
- enough GPU resources

This repo does **not** include:

- Nemotron base weights
- any private API key
- a guaranteed leaderboard-winning recipe

## What This Project Taught Me

If I had to summarize the project in one sentence, it would be:

**the hard part was not just making a notebook run, but turning scattered competition experiments into something that can be validated, reviewed, and shared.**

I intentionally kept the failures visible because they are often more useful than a single final score:

- why pure CoT can hurt
- why packing can break training quality
- why “merge then add a new LoRA” is not true continued training
- why output-format drift changes leaderboard results
- why small-sample CoT needs cleaning before anything else

## Related Files

- [record.md](./record.md)
- [docs/notebook_migration.md](./docs/notebook_migration.md)
- [notebooks/nvidia-nemotron-sfttrainer-training.ipynb](./notebooks/nvidia-nemotron-sfttrainer-training.ipynb)
- [notebooks/nvdia-nemotron-cot-sft.ipynb](./notebooks/nvdia-nemotron-cot-sft.ipynb)

## Acknowledgements

- Kaggle competition organizers
- the NVIDIA Nemotron team
- Dennis Fong for the public baseline notebook

If this repo helps you, feel free to fork it, open an issue, or use it as a starting point for your own competition engineering workflow.
