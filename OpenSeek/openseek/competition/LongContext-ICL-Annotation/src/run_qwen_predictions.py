"""Run Qwen3-4B predictions over all LongContext datasets and save JSONL outputs.

Output format per line:
{"id": "...", "prediction": "..."}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from method import build_prompt, build_prompt_with_reasoning, count_answer
from qwen_model import QwenModelLoader


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _iter_dataset_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("openseek-*.json"))
    if len(files) != 8:
        raise ValueError(f"Expected 8 dataset files, found {len(files)} in {data_dir}")
    return files


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _fit_prompt_to_budget(
    tokenizer,
    input_text: str,
    examples: list[dict[str, Any]],
    max_prompt_tokens: int,
    use_reasoning: bool = False,
) -> str:
    """Create an ICL prompt that fits token budget by reducing examples/truncating input."""
    prompt_builder = build_prompt_with_reasoning if use_reasoning else build_prompt
    
    for k in (3, 2, 1, 0):
        if use_reasoning:
            prompt = prompt_builder(input_text, examples[:k] if k > 0 else [])
        else:
            prompt = prompt_builder(input_text, examples[:k])
        if _token_count(tokenizer, prompt) <= max_prompt_tokens:
            return prompt

    # If prompt is still too long without examples, truncate the input text.
    minimal_prompt = prompt_builder("", []) if use_reasoning else build_prompt("", [])
    overhead = _token_count(tokenizer, minimal_prompt)
    available = max(64, max_prompt_tokens - overhead)
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    truncated_ids = input_ids[:available]
    truncated_input = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    return prompt_builder(truncated_input, []) if use_reasoning else build_prompt(truncated_input, [])


def _predict_from_prompt(model_loader: QwenModelLoader, prompt: str, max_new_tokens: int) -> str:
    raw = model_loader.generate_response(prompt, max_new_tokens=max_new_tokens)
    parsed = count_answer(raw)
    if parsed is None:
        return raw.strip()
    return str(parsed).strip()


def run_all_datasets(
    data_dir: Path,
    output_dir: Path,
    model_name_or_path: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    use_reasoning: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_files = _iter_dataset_files(data_dir)

    model_loader = QwenModelLoader(model_name_or_path=model_name_or_path)
    model_loader.load()

    for dataset_file in dataset_files:
        payload = _load_json(dataset_file)
        task_id = str(payload.get("task_id", dataset_file.stem))
        examples = payload.get("examples", [])
        test_samples = payload.get("test_samples", [])

        if not isinstance(examples, list):
            examples = []
        if not isinstance(test_samples, list):
            test_samples = []

        output_path = output_dir / f"{task_id}.jsonl"
        with output_path.open("w", encoding="utf-8") as writer:
            for sample in tqdm(test_samples, desc=f"Predicting {task_id}"):
                if not isinstance(sample, dict):
                    continue
                sample_id = sample.get("id")
                input_text = str(sample.get("input", ""))

                prompt = _fit_prompt_to_budget(
                    tokenizer=model_loader.tokenizer,
                    input_text=input_text,
                    examples=examples,
                    max_prompt_tokens=max_prompt_tokens,
                    use_reasoning=use_reasoning,
                )
                prediction = _predict_from_prompt(
                    model_loader=model_loader,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )

                writer.write(
                    json.dumps(
                        {"id": sample_id, "prediction": prediction},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-4B predictions for LongContext datasets")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Directory containing openseek-*.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "qwen_predictions",
        help="Directory to write JSONL predictions",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HuggingFace model id or local path",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=8192,
        help="Maximum prompt token budget before generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens for each sample",
    )
    parser.add_argument(
        "--use-reasoning",
        action="store_true",
        help="Enable multi-step reasoning mode with chain-of-thought prompting",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_datasets(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        use_reasoning=args.use_reasoning,
    )