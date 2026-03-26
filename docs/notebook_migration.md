# Notebook Migration Plan

This document maps the Kaggle notebook
`notebooks/nvidia-nemotron-sfttrainer-training.ipynb`
into a local project structure for Stage 2 dry runs and later server execution.

## What The Notebook Does

The notebook is a compact SFT baseline with five logical parts:

1. Install packages in Kaggle's offline environment.
2. Apply Kaggle-specific Triton and Nemotron runtime patches.
3. Download the base model and format `train.csv` into chat-style training text.
4. Load the model, attach LoRA, and run `SFTTrainer`.
5. Save the adapter and package `submission.zip`.

## Cell To File Mapping

### Cell 1: Offline package install

Notebook behavior:

```python
!pip install -q --no-index --find-links /kaggle/input/... datasets trl --ignore-installed
```

Local replacement:

- `requirements.txt`
- Optional bootstrap script:
  - `scripts/setup_env.ps1`
  - `scripts/setup_env.sh`

Notes:

- This is Kaggle-only installation logic and should not stay inside training code.
- Locally, we should install from PyPI or from a prepared wheel cache.

### Cell 4: Imports, Triton fixes, hyperparameters

Notebook behavior:

- imports core training dependencies
- patches `rmsnorm_fn`
- copies `ptxas-blackwell`
- hardcodes training hyperparameters

Local replacement:

- `train.py`
- optional `runtime_patches.py`
- `configs/sft_baseline.yaml`
- `configs/sft_cot.yaml`

Notes:

- The Triton/PTXAS patch is highly Kaggle-specific.
- We should isolate it behind a flag like `enable_kaggle_blackwell_patch`.
- Hyperparameters should move out of code and into config files.

### Cells 6-7: Model download, data loading, text formatting

Notebook behavior:

- downloads Nemotron via `kagglehub`
- reads competition `train.csv`
- samples 600 rows
- builds `text` with `tokenizer.apply_chat_template`

Local replacement:

- `data_utils.py`
- optional `download_model.py`
- `data/train.csv`

Recommended responsibilities for `data_utils.py`:

- load csv or jsonl
- optional subsample for quick dry runs
- split train/validation when needed
- format examples into chat text
- support both baseline data and CoT-augmented data

Notes:

- `SUBSAMPLE_SIZE = 600` should become a dry-run config option, not fixed code.
- The fallback ChatML string builder should stay because template support may vary.

### Cell 9: Stub imports, model loading, LoRA setup

Notebook behavior:

- inserts dummy modules to bypass missing `cutlass` and `mamba_ssm` imports
- loads the model
- disables Nemotron fast path
- attaches LoRA

Local replacement:

- `train.py`
- optional `model_utils.py`

Recommended responsibilities:

- load tokenizer
- load base model from a local path or HF path
- optionally apply compatibility stubs
- attach LoRA from config

Notes:

- The `cutlass` and `mamba_ssm` stubs may still be needed depending on the local environment.
- Keep them isolated so they are easy to turn on or remove.
- `device_map="auto"` is fine for GPU training but should be configurable for dry runs.

### Cell 11: SFT training

Notebook behavior:

- defines `SFTConfig`
- runs `SFTTrainer(...).train()`

Local replacement:

- `train.py`
- `configs/sft_baseline.yaml`
- `configs/sft_cot.yaml`

Recommended responsibilities:

- parse config
- instantiate `SFTConfig`
- build dataset
- train
- log effective config at startup

Notes:

- `bf16=True` will fail on unsupported hardware, so local dry-run config should allow:
  - `bf16: false`
  - `fp16: false`
  - `torch_dtype: float32`
- The dry run should use a tiny subset and maybe skip actual full model loading if needed.

### Cell 13: Save adapter and zip submission

Notebook behavior:

- saves LoRA adapter
- creates `submission.zip`
- checks `adapter_config.json` exists

Local replacement:

- `package_submission.py` or `scripts/package_submission.ps1`
- optional helper inside `train.py`

Recommended responsibilities:

- save adapter to `output/<run_name>/adapter/`
- create `output/<run_name>/submission.zip`
- verify required files exist before zipping

## Recommended Local Project Structure

```text
Nvidia/
  README.md
  requirements.txt
  .env.example
  .gitignore
  Dockerfile
  .dockerignore
  notebooks/
    nvidia-nemotron-sfttrainer-training.ipynb
  docs/
    notebook_migration.md
  configs/
    sft_baseline.yaml
    sft_cot.yaml
  data/
    README.md
    train.csv
    train_cot.jsonl
  prompts/
    cot_generation.txt
  scripts/
    setup_env.ps1
    run_train.ps1
    package_submission.ps1
  data_utils.py
  train.py
  generate_cot.py
  eval_local.py
  runtime_patches.py
  package_submission.py
```

## Minimum Files Needed For Stage 2 Dry Run

These are the files worth creating first:

1. `requirements.txt`
2. `.env.example`
3. `configs/sft_baseline.yaml`
4. `configs/sft_cot.yaml`
5. `data_utils.py`
6. `train.py`
7. `generate_cot.py`
8. `eval_local.py`
9. `package_submission.py`

Docker can wait until after these pass a local dry run.

## What Is Kaggle-Specific And Should Be Isolated

Keep these out of your main training flow as much as possible:

- offline `pip install --find-links /kaggle/input/...`
- `/kaggle/input/...` dataset paths
- `/kaggle/working/...` output paths
- Triton `ptxas-blackwell` copy logic
- Kaggle-only utility paths under `/kaggle/usr/lib/...`

If any of them are still needed, gate them behind config flags or environment detection.

## Stage 2 Additions Beyond The Notebook

The notebook only covers Stage 1 baseline SFT. For Stage 2 you still need:

- `generate_cot.py`
  - read baseline training samples
  - call an API model to generate reasoning traces
  - keep only rows whose boxed final answer matches ground truth
- `.env.example`
  - API key names
  - model name variables
  - output path variables
- `prompts/cot_generation.txt`
  - reusable CoT generation prompt template
- `eval_local.py`
  - boxed answer extraction
  - small local inference check
  - exact-match verification against labels

## Recommended Build Order

1. Convert notebook logic into `data_utils.py` and `train.py`.
2. Add baseline config and make sure a tiny dry run works locally.
3. Add `package_submission.py` and verify zip structure.
4. Add `.env.example`, `generate_cot.py`, and `prompts/cot_generation.txt`.
5. Add `eval_local.py` for boxed-answer checks.
6. Install Docker and write `Dockerfile`.

## Practical Dry-Run Advice

For the first local dry run, do not aim for real training quality. Aim to prove:

- config parsing works
- data formatting works
- tokenizer loads
- trainer starts
- output directory is created
- adapter saving and zipping work

Use a tiny setting such as:

- 8 to 32 samples
- 1 training step or 1 very short epoch
- no bf16 unless the GPU clearly supports it

## Biggest Risks To Watch

- local environment may not need Kaggle's Triton patch, but server environment might
- Nemotron remote code can have import-time assumptions that differ between machines
- `bf16` and Blackwell-specific logic may work in Kaggle but fail locally
- Stage 2 CoT generation will need a strict answer-validation step or the data will get noisy

## Bottom Line

The notebook should be split into:

- reusable training code
- config files
- environment-specific patches
- packaging utilities

Do not treat the notebook as the production training entrypoint. Use it as the source of truth for the baseline logic, then move the logic into scripts that can be dry-run, versioned, and containerized.
