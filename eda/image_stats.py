import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_io import discover_samples, load_image

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUTPUT_DIR = Path("output") / "eda"
DEFAULT_SPLIT = "train"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate image resolution/aspect ratio histograms and file size summary."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory that contains train/validation splits.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Split to analyze (e.g., train or validation).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write outputs.",
    )
    return parser.parse_args()


def summarize_sizes(values):
    arr = np.array(values, dtype=np.float64)
    return (
        int(arr.size),
        float(arr.mean()) if arr.size else 0.0,
        float(np.median(arr)) if arr.size else 0.0,
        int(arr.min()) if arr.size else 0,
        int(arr.max()) if arr.size else 0,
    )


def main():
    args = parse_args()
    samples = discover_samples(args.data_root, args.split)
    if not samples:
        print(f"No samples found under {args.data_root / args.split}")
        return

    widths = []
    heights = []
    areas = []
    ratios = []
    per_class_sizes = defaultdict(list)

    for sample in samples:
        image = load_image(sample.path)
        if image is None:
            print(f"Skipping image {sample.path}")
            continue
        height, width = image.shape[:2]
        widths.append(width)
        heights.append(height)
        areas.append(width * height)
        ratios.append(width / height if height else 0.0)
        per_class_sizes[sample.label].append(sample.path.stat().st_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / f"image_stats_{args.split}.png"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(areas, bins=30, color="#2c7fb8", alpha=0.8)
    axes[0].set_title("Resolution (width * height)")
    axes[0].set_xlabel("Pixels")
    axes[0].set_ylabel("Count")
    axes[1].hist(ratios, bins=30, color="#7fcdbb", alpha=0.8)
    axes[1].set_title("Aspect ratio (width / height)")
    axes[1].set_xlabel("Ratio")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    summary_path = args.output_dir / f"file_size_summary_{args.split}.csv"
    rows = [["label", "count", "mean_bytes", "median_bytes", "min_bytes", "max_bytes"]]
    for label in sorted(per_class_sizes):
        count, mean_b, med_b, min_b, max_b = summarize_sizes(per_class_sizes[label])
        rows.append([label, count, f"{mean_b:.2f}", f"{med_b:.2f}", min_b, max_b])

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    print(f"Wrote image stats plot to {plot_path}")
    print(f"Wrote file size summary to {summary_path}")


if __name__ == "__main__":
    main()
