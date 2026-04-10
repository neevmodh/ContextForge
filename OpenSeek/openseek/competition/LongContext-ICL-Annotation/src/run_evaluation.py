"""Demo script showing how to run evaluation."""

from pathlib import Path

from evaluate import main as evaluate_main


def main():
    """Run evaluation pipeline with default directories."""
    # Get paths relative to this script
    src_dir = Path(__file__).parent
    project_dir = src_dir.parent

    predictions_dir = project_dir / "outputs" / "qwen_predictions"
    data_dir = project_dir / "data"
    output_json = project_dir / "outputs" / "evaluation_results.json"

    print("🚀 Running Evaluation Pipeline\n")
    print(f"Predictions dir: {predictions_dir}")
    print(f"Data dir: {data_dir}")
    print(f"Output JSON: {output_json}\n")

    # Run evaluation
    results = evaluate_main(predictions_dir, data_dir, output_json)

    return results


if __name__ == "__main__":
    main()
