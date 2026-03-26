from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from data_utils import format_training_dataset, load_hf_dataset, maybe_split_dataset
from runtime_patches import disable_nemotron_fast_path, install_import_stubs, patch_rmsnorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nemotron local training entrypoint.")
    parser.add_argument(
        "--config",
        default="configs/sft_baseline.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset formatting before full training.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional override for dataset sample size.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    try:
        return dtype_map[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}") from exc


def resolve_model_path(model_cfg: dict[str, Any]) -> str:
    source = model_cfg.get("source", "huggingface")
    model_name_or_path = model_cfg["model_name_or_path"]

    if source == "kagglehub":
        import kagglehub

        return kagglehub.model_download(model_name_or_path)

    return model_name_or_path


def build_tokenizer(model_path: str, model_cfg: dict[str, Any]) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_lora_config(lora_cfg: dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )


def build_training_args(training_cfg: dict[str, Any]) -> SFTConfig:
    gradient_checkpointing = training_cfg.get("gradient_checkpointing", False)
    use_reentrant = training_cfg.get("gradient_checkpointing_use_reentrant", True)

    return SFTConfig(
        output_dir=training_cfg["output_dir"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        num_train_epochs=training_cfg["num_train_epochs"],
        learning_rate=training_cfg["learning_rate"],
        logging_steps=training_cfg["logging_steps"],
        bf16=training_cfg.get("bf16", False),
        fp16=training_cfg.get("fp16", False),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        optim=training_cfg.get("optim", "adamw_torch"),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.0),
        save_strategy=training_cfg.get("save_strategy", "no"),
        report_to=training_cfg.get("report_to", "none"),
        dataset_text_field="text",
        max_length=training_cfg["max_length"],
        packing=training_cfg.get("packing", False),
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
    )


def preview_dataset(dataset: Any) -> None:
    if len(dataset) == 0:
        print("Dataset is empty after formatting.")
        return

    sample_text = dataset[0]["text"]
    print(f"Formatted dataset size: {len(dataset)}")
    print("Formatted sample preview:")
    print(sample_text[:500])


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_cfg = dict(config["data"])
    model_cfg = dict(config["model"])
    runtime_cfg = dict(config.get("runtime", {}))
    training_cfg = dict(config["training"])

    if args.max_samples is not None:
        data_cfg["sample_size"] = args.max_samples
    elif args.dry_run and data_cfg.get("sample_size") is None:
        data_cfg["sample_size"] = 8

    output_dir = Path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    install_import_stubs(runtime_cfg.get("install_import_stubs", False))
    patch_rmsnorm()

    skip_model = args.dry_run and runtime_cfg.get("skip_model_load_in_dry_run", False)

    if not skip_model:
        model_path = resolve_model_path(model_cfg)
        print(f"Resolved model path: {model_path}")
        tokenizer = build_tokenizer(model_path, model_cfg)
    else:
        model_path = None
        tokenizer = None

    dataset = load_hf_dataset(
        data_cfg["path"],
        sample_size=data_cfg.get("sample_size"),
        seed=seed,
    )

    if skip_model:
        print(f"Dry run: loaded {len(dataset)} samples. Skipping tokenizer/model load.")
        return 0

    dataset = format_training_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        prompt_field=data_cfg.get("prompt_field", "prompt"),
        answer_field=data_cfg.get("answer_field", "answer"),
        reasoning_field=data_cfg.get("reasoning_field"),
        instruction_suffix=data_cfg.get(
            "instruction_suffix",
            "Put your final answer inside \\boxed{}.",
        ),
    )

    maybe_split = maybe_split_dataset(
        dataset,
        val_ratio=float(data_cfg.get("val_ratio", 0.0)),
        seed=seed,
    )

    if hasattr(maybe_split, "keys"):
        train_dataset = maybe_split["train"]
        eval_dataset = maybe_split["test"]
    else:
        train_dataset = maybe_split
        eval_dataset = None

    preview_dataset(train_dataset)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        torch_dtype=resolve_torch_dtype(model_cfg.get("torch_dtype", "bfloat16")),
    )

    disable_nemotron_fast_path(runtime_cfg.get("disable_fast_path", False))

    model = get_peft_model(model, build_lora_config(config["lora"]))
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=build_training_args(training_cfg),
    )

    print("Starting training...")
    trainer.train()
    trainer.model.save_pretrained(training_cfg["output_dir"])
    print(f"Adapter saved to {training_cfg['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
