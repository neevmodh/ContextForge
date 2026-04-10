#!/usr/bin/env python3
"""Explore OpenSeek LongContext-ICL-Annotation datasets.

This script:
1) Loads all JSON datasets in a directory
2) Prints dataset structure (fields + input/output format)
3) Shows 2 sample entries per dataset
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DatasetSummary:
    file_name: str
    task_id: str | None
    task_name: str | None
    top_level_fields: list[str]
    section_sizes: dict[str, int | None]
    input_type: str
    output_type: str
    sample_entries: list[dict[str, Any]]


def infer_type(value: Any) -> str:
    """Return a compact, human-readable type description for JSON-like data."""
    if isinstance(value, dict):
        keys = ", ".join(list(value.keys())[:5])
        suffix = "..." if len(value) > 5 else ""
        return f"dict(keys=[{keys}{suffix}])"
    if isinstance(value, list):
        if not value:
            return "list(empty)"
        return f"list[{infer_type(value[0])}]"
    if isinstance(value, str):
        return "str"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if value is None:
        return "null"
    return type(value).__name__


def compact_repr(value: Any, max_len: int = 220) -> str:
    """Render value as compact JSON-like text and truncate long strings."""
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def collect_samples(dataset: dict[str, Any], limit: int = 2) -> list[dict[str, Any]]:
    """Collect up to `limit` representative entries from known sample sections."""
    samples: list[dict[str, Any]] = []
    for section_name in ("examples", "test_samples"):
        section = dataset.get(section_name)
        if isinstance(section, list):
            for entry in section:
                if isinstance(entry, dict):
                    samples.append({"section": section_name, **entry})
                else:
                    samples.append({"section": section_name, "value": entry})
                if len(samples) >= limit:
                    return samples
    return samples


def infer_io_types(dataset: dict[str, Any]) -> tuple[str, str]:
    """Infer input and output types from the first available sample entry."""
    for section_name in ("examples", "test_samples"):
        section = dataset.get(section_name)
        if not isinstance(section, list) or not section:
            continue
        for entry in section:
            if not isinstance(entry, dict):
                continue
            input_type = infer_type(entry.get("input")) if "input" in entry else "missing"
            output_type = infer_type(entry.get("output")) if "output" in entry else "missing"
            return input_type, output_type
    return "unknown", "unknown"


def summarize_dataset(file_path: Path) -> DatasetSummary:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in {file_path.name}, got {type(data).__name__}")

    input_type, output_type = infer_io_types(data)
    section_sizes: dict[str, int | None] = {}
    for key in ("Definition", "examples", "test_samples"):
        value = data.get(key)
        section_sizes[key] = len(value) if isinstance(value, list) else None

    return DatasetSummary(
        file_name=file_path.name,
        task_id=data.get("task_id") if isinstance(data.get("task_id"), str) else None,
        task_name=data.get("task_name") if isinstance(data.get("task_name"), str) else None,
        top_level_fields=list(data.keys()),
        section_sizes=section_sizes,
        input_type=input_type,
        output_type=output_type,
        sample_entries=collect_samples(data, limit=2),
    )


def print_summary(summary: DatasetSummary) -> None:
    print("=" * 100)
    print(f"Dataset file : {summary.file_name}")
    print(f"Task ID      : {summary.task_id or 'N/A'}")
    print(f"Task name    : {summary.task_name or 'N/A'}")
    print(f"Fields       : {', '.join(summary.top_level_fields)}")
    print("Sections     : " + ", ".join(f"{k}={v if v is not None else 'N/A'}" for k, v in summary.section_sizes.items()))
    print(f"Input format : {summary.input_type}")
    print(f"Output format: {summary.output_type}")
    print("Sample entries:")

    if not summary.sample_entries:
        print("  - No sample entries found")
        return

    for idx, entry in enumerate(summary.sample_entries, start=1):
        section = entry.get("section", "unknown")
        sample_id = entry.get("id", "N/A")
        input_preview = compact_repr(entry.get("input", "<missing input>"))
        output_preview = compact_repr(entry.get("output", "<missing output>"))
        print(f"  [{idx}] section={section}, id={sample_id}")
        print(f"      input : {input_preview}")
        print(f"      output: {output_preview}")


def load_all_summaries(data_dir: Path) -> list[DatasetSummary]:
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    summaries: list[DatasetSummary] = []
    for file_path in json_files:
        summaries.append(summarize_dataset(file_path))
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore OpenSeek dataset JSON files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to folder containing dataset JSON files (default: script directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    summaries = load_all_summaries(data_dir)

    print(f"Found {len(summaries)} dataset files in: {data_dir}")
    for summary in summaries:
        print_summary(summary)


if __name__ == "__main__":
    main()