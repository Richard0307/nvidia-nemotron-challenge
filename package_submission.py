from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


MODEL_FILE_CANDIDATES = [
    "adapter_model.safetensors",
    "adapter_model.bin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package a LoRA adapter directory into Kaggle submission.zip.")
    parser.add_argument(
        "--adapter-dir",
        required=True,
        help="Directory containing the saved PEFT adapter files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output zip path. Defaults to <adapter-dir>/../submission.zip",
    )
    return parser.parse_args()


def validate_adapter_dir(adapter_dir: Path) -> list[Path]:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError("Missing adapter_config.json in adapter directory.")

    model_path = None
    for candidate in MODEL_FILE_CANDIDATES:
        candidate_path = adapter_dir / candidate
        if candidate_path.exists():
            model_path = candidate_path
            break

    if model_path is None:
        raise FileNotFoundError(
            "Missing adapter model weights. Expected adapter_model.safetensors or adapter_model.bin."
        )

    files = [path for path in adapter_dir.iterdir() if path.is_file()]
    return sorted(files)


def main() -> int:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    files = validate_adapter_dir(adapter_dir)

    output_path = Path(args.output) if args.output else adapter_dir.parent / "submission.zip"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zip_handle:
        for file_path in files:
            zip_handle.write(file_path, arcname=file_path.name)

    with zipfile.ZipFile(output_path, "r") as zip_handle:
        names = zip_handle.namelist()

    print(f"created={output_path}")
    print(f"files={names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
