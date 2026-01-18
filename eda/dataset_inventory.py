from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from src.data_io import discover_samples

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUTPUT_DIR = Path("output") / "eda"
DEFAULT_HEALTHY_LABELS = {"Healthy_Nail"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dataset inventory CSV with per-class and binary counts."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory that contains train/validation splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write CSV outputs.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional list of splits to include (defaults to all directories under data-root).",
    )
    parser.add_argument(
        "--healthy-labels",
        nargs="*",
        default=sorted(DEFAULT_HEALTHY_LABELS),
        help="Labels considered healthy for binary grouping.",
    )
    return parser.parse_args()


def discover_splits(data_root: Path, splits: list[str] | None) -> list[str]:
    if splits:
        return splits
    return sorted([path.name for path in data_root.iterdir() if path.is_dir()])


def main() -> None:
    args = parse_args()
    healthy_labels = set(args.healthy_labels)
    splits = discover_splits(args.data_root, args.splits)
    if not splits:
        raise RuntimeError(f"No split directories found under {args.data_root}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "dataset_inventory.csv"

    rows = [
        [
            "split",
            "group_type",
            "label",
            "count",
            "share_of_split",
            "is_healthy",
        ]
    ]

    for split in splits:
        samples = discover_samples(args.data_root, split)
        labels = [sample.label for sample in samples]
        total = len(labels)
        counts = Counter(labels)
        if total == 0:
            continue

        for label, count in sorted(counts.items()):
            is_healthy = label in healthy_labels
            rows.append(
                [
                    split,
                    "class",
                    label,
                    count,
                    f"{count / total:.4f}",
                    int(is_healthy),
                ]
            )

        healthy_count = sum(count for label, count in counts.items() if label in healthy_labels)
        non_healthy_count = total - healthy_count
        rows.append(
            [
                split,
                "binary",
                "healthy",
                healthy_count,
                f"{healthy_count / total:.4f}",
                1,
            ]
        )
        rows.append(
            [
                split,
                "binary",
                "non_healthy",
                non_healthy_count,
                f"{non_healthy_count / total:.4f}",
                0,
            ]
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    print(f"Wrote dataset inventory to {output_path}")


if __name__ == "__main__":
    main()
