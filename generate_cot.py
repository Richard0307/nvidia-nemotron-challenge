from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from data_utils import load_records


DEFAULT_SYSTEM_PROMPT = """You are generating supervised fine-tuning data for a reasoning model.
Given a puzzle-like prompt, infer the transformation rule from the examples, reason clearly, and end with the final answer inside \\boxed{}.
Do not mention that you know the ground-truth answer.
Keep the reasoning concise but useful for training."""


BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CoT data with an OpenAI-compatible API.")
    parser.add_argument("--input", default="data/train.csv", help="Input CSV or JSONL file.")
    parser.add_argument("--output", default="data/train_cot.jsonl", help="Output JSONL path.")
    parser.add_argument("--env-file", default=".env", help="Path to an env file.")
    parser.add_argument("--prompt-field", default="prompt", help="Input prompt field.")
    parser.add_argument("--answer-field", default="answer", help="Ground-truth answer field.")
    parser.add_argument("--id-field", default="id", help="Record id field.")
    parser.add_argument(
        "--system-prompt-file",
        default="prompts/cot_generation.txt",
        help="Optional prompt template file.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap for dry runs.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Delay between API calls.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists.")
    return parser.parse_args()


def load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def load_system_prompt(path: str | Path) -> str:
    prompt_path = Path(path)
    if not prompt_path.exists():
        return DEFAULT_SYSTEM_PROMPT
    return prompt_path.read_text(encoding="utf-8").strip()


def extract_boxed_answer(text: str) -> str | None:
    matches = BOXED_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(text: str) -> str:
    text = str(text).strip()
    # Strip LaTeX space escapes: "\ " → " ", then remove stray backslashes
    text = re.sub(r"\\ ", " ", text)
    text = text.replace("\\", "")
    return " ".join(text.split())


def answers_match(expected: str, predicted: str | None) -> bool:
    if predicted is None:
        return False
    return normalize_answer(expected) == normalize_answer(predicted)


def build_user_prompt(problem_prompt: str) -> str:
    return (
        "Solve the following reasoning task.\n"
        "Infer the rule from the examples, explain the reasoning clearly, and end with the final answer inside \\boxed{}.\n\n"
        f"{problem_prompt}"
    )


def call_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    system_prompt: str,
    user_prompt: str,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    is_o_series = re.match(r"^o\d", model) is not None
    tokens_key = "max_completion_tokens" if is_o_series else "max_tokens"
    payload = {
        "model": model,
        tokens_key: max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if not is_o_series:
        payload["temperature"] = temperature

    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with request.urlopen(req, timeout=timeout_seconds) as response:
        data = json.loads(response.read().decode("utf-8"))

    message = data["choices"][0]["message"]
    # o-series reasoning models may return content as None; fall back to empty string
    content = message.get("content") or ""
    if not content:
        # Print raw message for debugging
        print(f"    [DEBUG] raw message keys: {list(message.keys())}")
        print(f"    [DEBUG] raw message: {str(message)[:500]}")
    return content


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    if output_path.exists() and args.overwrite:
        output_path.unlink()

    api_key = get_required_env("COT_API_KEY")
    base_url = os.environ.get("COT_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("COT_MODEL", "gpt-4.1-mini")
    temperature = float(os.environ.get("COT_TEMPERATURE", "0.2"))
    max_tokens = int(os.environ.get("COT_MAX_TOKENS", "1200"))
    timeout_seconds = int(os.environ.get("COT_TIMEOUT_SECONDS", "120"))
    system_prompt = load_system_prompt(args.system_prompt_file)

    records = load_records(args.input)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    accepted = 0
    rejected = 0

    for index, record in enumerate(records, start=1):
        problem_prompt = str(record[args.prompt_field])
        ground_truth = str(record[args.answer_field]).strip()

        try:
            content = call_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                system_prompt=system_prompt,
                user_prompt=build_user_prompt(problem_prompt),
            )
        except error.HTTPError as exc:
            rejected += 1
            body = exc.read().decode("utf-8", errors="ignore")
            print(f"[{index}] HTTP error {exc.code}: {body[:300]}")
            continue
        except Exception as exc:
            rejected += 1
            print(f"[{index}] Request failed: {exc}")
            continue

        boxed_answer = extract_boxed_answer(content)
        is_match = answers_match(ground_truth, boxed_answer)

        if is_match:
            row = {
                "id": record.get(args.id_field, index),
                "prompt": problem_prompt,
                "answer": ground_truth,
                "reasoning": content,
                "boxed_answer": boxed_answer,
                "source_model": model,
            }
            write_jsonl(output_path, [row])
            accepted += 1
            print(f"[{index}] accepted")
        else:
            rejected += 1
            preview = content[:300].replace("\n", " ") if content else "(empty)"
            print(
                f"[{index}] rejected | expected={ground_truth!r} | predicted={boxed_answer!r}"
            )
            print(f"    model output preview: {preview}")

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    print(
        f"Finished. accepted={accepted} rejected={rejected} output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
