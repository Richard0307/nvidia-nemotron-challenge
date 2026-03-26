from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from datasets import Dataset, DatasetDict


def load_records(data_path: str | Path) -> list[dict[str, Any]]:
    path = Path(data_path)
    suffix = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if suffix == ".csv":
        frame = pl.read_csv(path)
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pl.read_ndjson(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    return frame.to_dicts()


def load_hf_dataset(
    data_path: str | Path,
    sample_size: int | None = None,
    seed: int = 42,
) -> Dataset:
    records = load_records(data_path)
    dataset = Dataset.from_list(records)

    if sample_size is not None:
        sample_size = min(sample_size, len(dataset))
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    return dataset


def build_training_text(
    example: dict[str, Any],
    tokenizer: Any,
    prompt_field: str = "prompt",
    answer_field: str = "answer",
    reasoning_field: str | None = None,
    instruction_suffix: str = "Put your final answer inside \\boxed{}.",
) -> dict[str, str]:
    prompt = str(example[prompt_field]).strip()
    answer = str(example[answer_field]).strip()
    reasoning = None

    if reasoning_field:
        raw_reasoning = example.get(reasoning_field)
        if raw_reasoning is not None:
            reasoning = str(raw_reasoning).strip()

    user_msg = prompt
    if instruction_suffix:
        user_msg = f"{prompt}\n{instruction_suffix}"

    if reasoning:
        assistant_msg = reasoning
        if "\\boxed{" not in assistant_msg:
            assistant_msg = f"{assistant_msg}\n\nTherefore, the answer is \\boxed{{{answer}}}"
    else:
        assistant_msg = f"\\boxed{{{answer}}}"

    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        text = (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        )

    return {"text": text}


def format_training_dataset(
    dataset: Dataset,
    tokenizer: Any,
    prompt_field: str = "prompt",
    answer_field: str = "answer",
    reasoning_field: str | None = None,
    instruction_suffix: str = "Put your final answer inside \\boxed{}.",
) -> Dataset:
    source_columns = list(dataset.column_names)
    return dataset.map(
        build_training_text,
        fn_kwargs={
            "tokenizer": tokenizer,
            "prompt_field": prompt_field,
            "answer_field": answer_field,
            "reasoning_field": reasoning_field,
            "instruction_suffix": instruction_suffix,
        },
        remove_columns=source_columns,
    )


def maybe_split_dataset(
    dataset: Dataset,
    val_ratio: float = 0.0,
    seed: int = 42,
) -> Dataset | DatasetDict:
    if val_ratio <= 0:
        return dataset

    return dataset.train_test_split(test_size=val_ratio, seed=seed)
