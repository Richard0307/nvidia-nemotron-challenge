from __future__ import annotations

import argparse
import math
import re
from typing import Any

from data_utils import load_records


BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate boxed answers from a local predictions file.")
    parser.add_argument("--input", required=True, help="CSV or JSONL file containing predictions.")
    parser.add_argument("--answer-field", default="answer", help="Ground-truth answer field.")
    parser.add_argument(
        "--prediction-field",
        default="prediction",
        help="Field containing the raw model output or predicted answer.",
    )
    parser.add_argument(
        "--numeric-tolerance",
        type=float,
        default=None,
        help="Optional numeric tolerance for float-like answers.",
    )
    parser.add_argument("--show-misses", type=int, default=5, help="How many misses to print.")
    return parser.parse_args()


def extract_boxed_answer(text: str) -> str | None:
    matches = BOXED_PATTERN.findall(str(text))
    if not matches:
        return None
    return matches[-1].strip()


def normalize_answer(text: Any) -> str:
    return " ".join(str(text).strip().split())


def maybe_to_float(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def answers_match(expected: str, predicted: str | None, numeric_tolerance: float | None) -> bool:
    if predicted is None:
        return False

    normalized_expected = normalize_answer(expected)
    normalized_predicted = normalize_answer(predicted)

    if normalized_expected == normalized_predicted:
        return True

    if numeric_tolerance is None:
        return False

    expected_value = maybe_to_float(normalized_expected)
    predicted_value = maybe_to_float(normalized_predicted)

    if expected_value is None or predicted_value is None:
        return False

    return math.isclose(expected_value, predicted_value, abs_tol=numeric_tolerance)


def main() -> int:
    args = parse_args()
    rows = load_records(args.input)

    total = 0
    matched = 0
    misses: list[dict[str, str | None]] = []

    for row in rows:
        total += 1
        expected = str(row[args.answer_field]).strip()
        raw_prediction = str(row[args.prediction_field])
        boxed_prediction = extract_boxed_answer(raw_prediction)
        predicted = boxed_prediction if boxed_prediction is not None else raw_prediction.strip()

        if answers_match(expected, predicted, args.numeric_tolerance):
            matched += 1
        elif len(misses) < args.show_misses:
            misses.append(
                {
                    "expected": expected,
                    "predicted": predicted,
                    "boxed_prediction": boxed_prediction,
                }
            )

    accuracy = matched / total if total else 0.0
    print(f"evaluated={total}")
    print(f"matched={matched}")
    print(f"accuracy={accuracy:.4f}")

    if misses:
        print("sample misses:")
        for miss in misses:
            print(miss)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
