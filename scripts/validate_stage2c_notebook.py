from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path


SEED = 42
DIRECT_SAMPLES = 5000
COT_OVERSAMPLE = 2

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
BOXED_TAIL_RE = re.compile(r"\\boxed\{[^{}]+\}\s*$")


def normalize_answer(text: str) -> str:
    text = str(text).strip().replace(r"\ ", " ")
    return " ".join(text.split())


def extract_last_boxed(text: str) -> str | None:
    matches = BOXED_RE.findall(str(text))
    if not matches:
        return None
    return normalize_answer(matches[-1])


def strip_trailing_boxed(reasoning: str) -> str:
    reasoning = str(reasoning).strip().replace(r"\ ", " ")
    last_idx = reasoning.rfind(r"\boxed{")
    if last_idx != -1:
        reasoning = reasoning[:last_idx].rstrip(" .。．")
    return reasoning.strip()


def make_text(user_msg: str, assistant_msg: str) -> str:
    return (
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
    )


def build_direct_text(row: dict[str, str]) -> str:
    prompt = str(row["prompt"]).strip()
    answer = normalize_answer(row["answer"])
    user_msg = prompt + "\nPut your final answer inside \\boxed{}."
    assistant_msg = f"\\boxed{{{answer}}}"
    return make_text(user_msg, assistant_msg)


def build_cot_text(row: dict[str, str]) -> str:
    prompt = row["prompt"]
    reasoning = row["reasoning"]
    answer = row["answer"]
    user_msg = prompt + "\nPut your final answer inside \\boxed{}."
    assistant_msg = f"{reasoning}\n\n\\boxed{{{answer}}}"
    return make_text(user_msg, assistant_msg)


def load_train_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_cot_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the revised Stage 2c notebook data logic.")
    parser.add_argument(
        "--train",
        default="data/train.csv",
        help="Local train.csv path.",
    )
    parser.add_argument(
        "--cot",
        default="data/train_cot.jsonl",
        help="Local train_cot.jsonl path.",
    )
    parser.add_argument(
        "--write-cleaned",
        default="",
        help="Optional cleaned CoT output JSONL path.",
    )
    args = parser.parse_args()

    train_rows = load_train_rows(Path(args.train))
    cot_raw_rows = load_cot_rows(Path(args.cot))

    cleaned_cot_rows: list[dict[str, str]] = []
    rejected = 0
    tail_boxed_count = 0
    boxed_any_count = 0
    backslash_space_count = 0

    for row in cot_raw_rows:
        prompt = str(row.get("prompt", "")).strip()
        answer = normalize_answer(row.get("answer", ""))
        reasoning = str(row.get("reasoning", "")).strip()
        boxed = extract_last_boxed(reasoning)

        if r"\boxed{" in reasoning:
            boxed_any_count += 1
        if BOXED_TAIL_RE.search(reasoning):
            tail_boxed_count += 1
        if r"\ " in reasoning:
            backslash_space_count += 1

        if not prompt or not answer or not reasoning or boxed != answer:
            rejected += 1
            continue

        clean_reasoning = strip_trailing_boxed(reasoning)
        if not clean_reasoning:
            rejected += 1
            continue

        cleaned_cot_rows.append(
            {
                "id": row.get("id", ""),
                "prompt": prompt,
                "answer": answer,
                "reasoning": clean_reasoning,
                "boxed_answer": boxed,
            }
        )

    cot_ids = {row["id"] for row in cleaned_cot_rows if row.get("id")}
    filtered_train_rows = [row for row in train_rows if row.get("id") not in cot_ids]

    random.seed(SEED)
    direct_count = min(DIRECT_SAMPLES, len(filtered_train_rows))
    direct_rows = random.sample(filtered_train_rows, direct_count)

    direct_texts = [build_direct_text(row) for row in direct_rows]
    cot_texts = [build_cot_text(row) for row in cleaned_cot_rows for _ in range(COT_OVERSAMPLE)]

    def assistant_boxed_count(text: str) -> int:
        marker = "<|im_start|>assistant\n"
        assistant_part = str(text).split(marker, 1)[-1]
        return assistant_part.count("\\boxed{")

    direct_single_boxed = sum(assistant_boxed_count(text) == 1 for text in direct_texts)
    cot_single_boxed = sum(assistant_boxed_count(text) == 1 for text in cot_texts)
    cot_with_think = sum("<think>" in text for text in cot_texts)
    direct_with_think = sum("<think>" in text for text in direct_texts)

    print(f"train_rows={len(train_rows)}")
    print(f"cot_raw_rows={len(cot_raw_rows)}")
    print(f"cot_boxed_any={boxed_any_count}")
    print(f"cot_tail_boxed={tail_boxed_count}")
    print(f"cot_backslash_space={backslash_space_count}")
    print(f"cot_clean_rows={len(cleaned_cot_rows)}")
    print(f"cot_rejected={rejected}")
    print(f"train_rows_after_dedup={len(filtered_train_rows)}")
    print(f"direct_rows_sampled={len(direct_rows)}")
    print(f"cot_rows_after_oversample={len(cot_texts)}")
    print(f"mixed_total_rows={len(direct_texts) + len(cot_texts)}")
    print(f"estimated_optimizer_steps={(len(direct_texts) + len(cot_texts)) // 4}")
    print(f"direct_single_boxed={direct_single_boxed}/{len(direct_texts)}")
    print(f"cot_single_boxed={cot_single_boxed}/{len(cot_texts)}")
    print(f"direct_with_think={direct_with_think}")
    print(f"cot_with_think={cot_with_think}")

    assert len(cleaned_cot_rows) > 0, "No cleaned CoT rows survived."
    assert direct_single_boxed == len(direct_texts), "Some direct samples do not contain exactly one boxed answer."
    assert cot_single_boxed == len(cot_texts), "Some CoT samples do not contain exactly one boxed answer."
    assert direct_with_think == 0, "Direct samples still contain <think> tags."
    assert cot_with_think == 0, "CoT samples still contain <think> tags."
    assert direct_count == min(DIRECT_SAMPLES, len(filtered_train_rows)), "Unexpected direct sample count."

    if args.write_cleaned:
        output_path = Path(args.write_cleaned)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in cleaned_cot_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"cleaned_cot_written={output_path}")

    print("\nDirect preview:")
    print(direct_texts[0][:400])
    print("\nCoT preview:")
    print(cot_texts[0][:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
