"""Evaluate predictions against ground truth and generate accuracy reports.

This module compares JSONL predictions with ground truth labels and calculates
per-dataset and overall accuracy metrics.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_predictions(predictions_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load all JSONL prediction files from directory.

    Args:
        predictions_dir: Directory containing task_id.jsonl files

    Returns:
        Dict mapping task_id to list of prediction records
    """
    predictions = {}
    jsonl_files = sorted(predictions_dir.glob("*.jsonl"))

    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {predictions_dir}")

    for jsonl_file in jsonl_files:
        task_id = jsonl_file.stem
        records = []
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        predictions[task_id] = records

    return predictions


def load_ground_truth(data_dir: Path) -> dict[str, dict[str, str]]:
    """Load ground truth from dataset files.

    Looks for openseek-*.json files and attempts to extract ground truth
    from test_samples (if available) or creates a mapping from examples.

    Args:
        data_dir: Directory containing openseek-*.json dataset files

    Returns:
        Dict mapping task_id -> {sample_id -> expected_output}
    """
    ground_truth = {}
    dataset_files = sorted(data_dir.glob("openseek-*.json"))

    if not dataset_files:
        raise ValueError(f"No openseek-*.json files found in {data_dir}")

    for dataset_file in dataset_files:
        with dataset_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        task_id = data.get("task_id", dataset_file.stem)
        gt_map = {}

        # First, try to use ground truth from test_samples
        test_samples = data.get("test_samples", [])
        for sample in test_samples:
            if isinstance(sample, dict) and "id" in sample and "output" in sample:
                output = sample["output"]
                # Output might be list, convert to string
                gt_map[sample["id"]] = (
                    str(output[0]) if isinstance(output, list) and output else str(output)
                )

        # If no test_samples have ground truth, use examples as reference
        if not gt_map:
            examples = data.get("examples", [])
            for example in examples[:100]:  # Use first 100 examples
                if isinstance(example, dict) and "id" in example and "output" in example:
                    output = example["output"]
                    gt_map[example["id"]] = (
                        str(output[0])
                        if isinstance(output, list) and output
                        else str(output)
                    )

        if gt_map:
            ground_truth[task_id] = gt_map

    return ground_truth


def normalize_output(output: Any) -> str:
    """Normalize output for comparison (case-insensitive, whitespace trimmed)."""
    if isinstance(output, (list, tuple)):
        output = output[0] if output else ""
    output_str = str(output).strip().lower()
    return output_str


def calculate_accuracy(
    predictions: dict[str, list[dict[str, Any]]],
    ground_truth: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Calculate accuracy metrics comparing predictions with ground truth.

    Args:
        predictions: Dict mapping task_id to list of prediction records
        ground_truth: Dict mapping task_id to id->expected_output

    Returns:
        Dict containing per-dataset and overall accuracy metrics
    """
    results = {
        "per_dataset": {},
        "overall": {},
        "missing_gt": [],
    }

    total_correct = 0
    total_samples = 0

    for task_id, preds in predictions.items():
        gt_map = ground_truth.get(task_id, {})

        if not gt_map:
            results["missing_gt"].append(task_id)
            continue

        correct = 0
        matched = 0
        for pred in preds:
            sample_id = pred.get("id")
            prediction = pred.get("prediction", "")
            expected = gt_map.get(sample_id)

            if expected is None:
                continue

            matched += 1
            pred_normalized = normalize_output(prediction)
            expected_normalized = normalize_output(expected)

            if pred_normalized == expected_normalized:
                correct += 1

        if matched > 0:
            accuracy = correct / matched
            results["per_dataset"][task_id] = {
                "correct": correct,
                "total": matched,
                "accuracy": accuracy,
            }
            total_correct += correct
            total_samples += matched

    # Calculate overall accuracy
    if total_samples > 0:
        results["overall"] = {
            "correct": total_correct,
            "total": total_samples,
            "accuracy": total_correct / total_samples,
        }
    else:
        results["overall"] = {"correct": 0, "total": 0, "accuracy": 0.0}

    return results


def print_results(
    results: dict[str, Any],
    sort_by: str = "accuracy",
) -> None:
    """Print evaluation results in a readable format.

    Args:
        results: Accuracy metrics from calculate_accuracy()
        sort_by: Column to sort by ('accuracy', 'correct', 'total')
    """
    print("\n" + "=" * 90)
    print("EVALUATION RESULTS".center(90))
    print("=" * 90)

    # Per-dataset results
    if results["per_dataset"]:
        print("\n📊 Per-Dataset Accuracy:\n")

        # Prepare data
        rows = []
        for task_id, metrics in results["per_dataset"].items():
            rows.append(
                {
                    "task_id": task_id,
                    "correct": metrics["correct"],
                    "total": metrics["total"],
                    "accuracy": metrics["accuracy"],
                }
            )

        # Sort by accuracy (descending) by default
        if sort_by == "accuracy":
            rows.sort(key=lambda x: x["accuracy"], reverse=True)
        elif sort_by == "correct":
            rows.sort(key=lambda x: x["correct"], reverse=True)
        elif sort_by == "total":
            rows.sort(key=lambda x: x["total"], reverse=True)

        # Print header
        print(f"{'Task ID':<35} {'Correct':>10} {'Total':>10} {'Accuracy':>12}")
        print("-" * 90)

        # Print rows
        for row in rows:
            print(
                f"{row['task_id']:<35} {row['correct']:>10} {row['total']:>10} {row['accuracy']:>11.1%}"
            )

    # Overall results
    overall = results["overall"]
    if overall["total"] > 0:
        print("\n" + "-" * 90)
        print(f"\n✅ Overall Accuracy: {overall['accuracy']:.1%}")
        print(f"   Correct: {overall['correct']} / {overall['total']}")
        print()

    # Warnings
    if results["missing_gt"]:
        print(f"\n⚠️  Ground truth not found for: {', '.join(results['missing_gt'])}")
        print("   (Skipped in accuracy calculation)")
    
    print("=" * 90 + "\n")


def export_results(
    results: dict[str, Any],
    output_file: Path,
) -> None:
    """Export results to JSON file for further analysis.

    Args:
        results: Accuracy metrics from calculate_accuracy()
        output_file: Path to save JSON results
    """
    # Convert numpy types to native Python types for JSON serialization
    export_data = {
        "overall": {
            "correct": int(results["overall"]["correct"]),
            "total": int(results["overall"]["total"]),
            "accuracy": float(results["overall"]["accuracy"]),
        },
        "per_dataset": {
            task_id: {
                "correct": int(m["correct"]),
                "total": int(m["total"]),
                "accuracy": float(m["accuracy"]),
            }
            for task_id, m in results["per_dataset"].items()
        },
        "missing_gt": results["missing_gt"],
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Results exported to {output_file}")


def main(
    predictions_dir: Path,
    data_dir: Path,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Run full evaluation pipeline.

    Args:
        predictions_dir: Directory with JSONL predictions
        data_dir: Directory with openseek-*.json datasets
        output_json: Optional path to save results JSON

    Returns:
        Results dict from calculate_accuracy()
    """
    print("Loading predictions...")
    predictions = load_predictions(predictions_dir)
    print(f"  Loaded {len(predictions)} task predictions")

    print("Loading ground truth...")
    ground_truth = load_ground_truth(data_dir)
    print(f"  Loaded ground truth for {len(ground_truth)} tasks")

    print("\nCalculating accuracy...")
    results = calculate_accuracy(predictions, ground_truth)

    print_results(results)

    if output_json:
        export_results(results, output_json)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate predictions against ground truth"
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path(".") / "outputs" / "qwen_predictions",
        help="Directory containing JSONL predictions",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".") / "data",
        help="Directory containing openseek-*.json datasets",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(".") / "outputs" / "evaluation_results.json",
        help="Optional path to save results as JSON",
    )

    args = parser.parse_args()
    main(args.predictions_dir, args.data_dir, args.output_json)
