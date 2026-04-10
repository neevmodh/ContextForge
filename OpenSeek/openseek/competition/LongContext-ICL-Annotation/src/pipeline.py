"""End-to-end pipeline: generate predictions and evaluate in one command."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluate import calculate_accuracy, load_ground_truth, load_predictions, print_results
from run_qwen_predictions import parse_args as parse_prediction_args
from run_qwen_predictions import run_all_datasets


def main(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    model_name_or_path: str = "Qwen/Qwen3-4B",
    max_prompt_tokens: int = 8192,
    max_new_tokens: int = 256,
    use_reasoning: bool = False,
    use_self_consistency: bool = False,
    skip_prediction: bool = False,
) -> None:
    """Run full pipeline: predict then evaluate.

    Args:
        data_dir: Directory with datasets
        output_dir: Directory to save predictions
        model_name_or_path: Model identifier
        max_prompt_tokens: Token budget for prompts
        max_new_tokens: Max tokens to generate
        use_reasoning: Enable chain-of-thought reasoning
        use_self_consistency: Enable voting (3x runs)
        skip_prediction: Skip prediction and only evaluate existing results
    """
    # Set defaults
    src_dir = Path(__file__).parent
    project_dir = src_dir.parent

    if data_dir is None:
        data_dir = project_dir / "data"
    if output_dir is None:
        output_dir = project_dir / "outputs" / "qwen_predictions"

    print("\n" + "=" * 90)
    print("🚀 LongContext-ICL Annotation Pipeline".center(90))
    print("=" * 90)

    # Step 1: Generate predictions
    if not skip_prediction:
        print("\n📝 Step 1: Generating Predictions")
        print("-" * 90)
        print(f"  Model: {model_name_or_path}")
        print(f"  Reasoning: {'✓' if use_reasoning else '✗'}")
        print(f"  Self-Consistency: {'✓' if use_self_consistency else '✗'}")
        print(f"  Output dir: {output_dir}\n")

        run_all_datasets(
            data_dir=data_dir,
            output_dir=output_dir,
            model_name_or_path=model_name_or_path,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            use_reasoning=use_reasoning,
            use_self_consistency=use_self_consistency,
        )
        print("✓ Predictions completed")
    else:
        print("\n⏭️  Skipping prediction step (using existing results)")

    # Step 2: Evaluate predictions
    print("\n📊 Step 2: Evaluating Results")
    print("-" * 90)

    try:
        print("  Loading predictions...")
        predictions = load_predictions(output_dir)
        print(f"    ✓ Loaded {len(predictions)} task results")

        print("  Loading ground truth...")
        ground_truth = load_ground_truth(data_dir)
        print(f"    ✓ Loaded ground truth for {len(ground_truth)} tasks")

        print("  Calculating accuracy...")
        results = calculate_accuracy(predictions, ground_truth)

        # Print results
        print_results(results)

        # Save results JSON
        results_json = output_dir / "accuracy_results.json"
        import json

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

        output_dir.mkdir(parents=True, exist_ok=True)
        with results_json.open("w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Results saved to {results_json}")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        print("   (This may be normal if ground truth is not available)")

    print("\n" + "=" * 90)
    print("✅ Pipeline Complete".center(90))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: predict and evaluate LongContext datasets"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing openseek-*.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
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
    parser.add_argument(
        "--use-self-consistency",
        action="store_true",
        help="Enable self-consistency voting: run model 3x per input and majority vote",
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help="Skip prediction generation and only evaluate existing results",
    )

    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name_or_path=args.model_name_or_path,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        use_reasoning=args.use_reasoning,
        use_self_consistency=args.use_self_consistency,
        skip_prediction=args.skip_prediction,
    )
