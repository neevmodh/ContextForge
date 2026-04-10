"""Dataset loading utilities for LongContext-ICL-Annotation.

This module provides a reusable DatasetLoader that reads all task JSON files
from the data folder and converts records into a unified format:

{
    "input": ...,
    "label": ...,
    "examples": ...,
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DatasetLoader:
    """Load and normalize LongContext task datasets."""

    def __init__(self, data_dir: str | Path | None = None, example_limit: int | None = None) -> None:
        base_dir = Path(__file__).resolve().parent
        self.data_dir = Path(data_dir) if data_dir is not None else (base_dir.parent / "data")
        self.example_limit = example_limit

    def load_all_datasets(self) -> dict[str, list[dict[str, Any]]]:
        """Load every openseek task file and return unified records keyed by task id.

        Returns:
            A dictionary mapping task id (for example, "openseek-1") to a list
            of unified records. Records are built from both `examples` and
            `test_samples` sections when available.
        """
        dataset_files = self._discover_dataset_files()
        all_data: dict[str, list[dict[str, Any]]] = {}

        for dataset_file in dataset_files:
            task_payload = self._load_json(dataset_file)
            task_id = str(task_payload.get("task_id") or dataset_file.stem)
            all_data[task_id] = self._normalize_task_payload(task_payload)

        return all_data

    def load_dataset(self, task_id: str) -> list[dict[str, Any]]:
        """Load one dataset by task id, for example: openseek-3."""
        all_datasets = self.load_all_datasets()
        if task_id not in all_datasets:
            available = ", ".join(sorted(all_datasets.keys()))
            raise KeyError(f"Task '{task_id}' not found. Available tasks: {available}")
        return all_datasets[task_id]

    def _discover_dataset_files(self) -> list[Path]:
        files = sorted(self.data_dir.glob("openseek-*.json"))
        if not files:
            raise FileNotFoundError(f"No dataset files found in: {self.data_dir}")
        return files

    def _load_json(self, file_path: Path) -> dict[str, Any]:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object at top-level in {file_path.name}")
        return data

    def _normalize_task_payload(self, task_payload: dict[str, Any]) -> list[dict[str, Any]]:
        examples_bank = self._build_examples_bank(task_payload.get("examples"))
        records: list[dict[str, Any]] = []

        records.extend(self._normalize_split(task_payload.get("examples"), examples_bank))
        records.extend(self._normalize_split(task_payload.get("test_samples"), examples_bank))
        return records

    def _build_examples_bank(self, raw_examples: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_examples, list):
            return []

        normalized_examples: list[dict[str, Any]] = []
        for item in raw_examples:
            if not isinstance(item, dict):
                continue
            normalized_examples.append(
                {
                    "input": item.get("input"),
                    "label": self._extract_label(item),
                }
            )

        if self.example_limit is None:
            return normalized_examples
        return normalized_examples[: self.example_limit]

    def _normalize_split(self, split_data: Any, examples_bank: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(split_data, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in split_data:
            if not isinstance(item, dict):
                continue

            normalized.append(
                {
                    "input": item.get("input"),
                    "label": self._extract_label(item),
                    "examples": examples_bank,
                }
            )

        return normalized

    def _extract_label(self, item: dict[str, Any]) -> Any:
        """Extract labels across heterogeneous field names and output shapes."""
        for key in ("label", "labels", "target", "targets", "output", "outputs", "answer"):
            if key in item:
                return self._normalize_label_value(item[key])
        return None

    @staticmethod
    def _normalize_label_value(value: Any) -> Any:
        # Most tasks store output as a one-element list. Convert this to scalar
        # for a consistent and convenient label field.
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value


if __name__ == "__main__":
    loader = DatasetLoader()
    datasets = loader.load_all_datasets()
    for task, rows in sorted(datasets.items()):
        print(f"{task}: {len(rows)} unified samples")