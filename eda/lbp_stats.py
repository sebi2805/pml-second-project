from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern

from src.data_io import discover_samples, load_image

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUTPUT_DIR = Path("output") / "eda"
DEFAULT_SPLIT = "train"
DEFAULT_RESIZE = (128, 128)
DEFAULT_RADIUS = 1
DEFAULT_POINTS = 8
DEFAULT_METHOD = "uniform"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mean LBP histograms per class and compare them."
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
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=list(DEFAULT_RESIZE),
        metavar=("WIDTH", "HEIGHT"),
        help="Resize images for LBP extraction.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_RADIUS,
        help="LBP radius.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=DEFAULT_POINTS,
        help="Number of LBP sampling points.",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=("uniform", "default", "ror"),
        help="LBP method (uniform, default, ror).",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=0,
        help="Optional cap per class (0 means no cap).",
    )
    return parser.parse_args()


def lbp_num_bins(n_points: int, method: str) -> int:
    if method == "uniform":
        return n_points + 2
    if method in ("default", "ror"):
        return 2**n_points
    raise ValueError(f"Unsupported LBP method: {method}")


def compute_lbp_hist(
    image_bgr: np.ndarray, n_points: int, radius: float, method: str, n_bins: int
) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method=method)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float64)
    hist /= hist.sum() + 1e-8
    return hist


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def main() -> None:
    args = parse_args()
    samples = discover_samples(args.data_root, args.split)
    if not samples:
        raise RuntimeError(f"No samples found under {args.data_root / args.split}")

    resize = tuple(args.resize)
    n_bins = lbp_num_bins(args.n_points, args.method)

    per_class_samples: dict[str, list[Path]] = defaultdict(list)
    for sample in samples:
        per_class_samples[sample.label].append(sample.path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    lbp_means: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for label in sorted(per_class_samples):
        paths = per_class_samples[label]
        if args.max_images_per_class > 0:
            paths = paths[: args.max_images_per_class]

        sum_hist = np.zeros(n_bins, dtype=np.float64)
        count = 0
        for path in paths:
            image = load_image(path, resize=resize)
            hist = compute_lbp_hist(
                image, n_points=args.n_points, radius=args.radius, method=args.method, n_bins=n_bins
            )
            sum_hist += hist
            count += 1

        if count == 0:
            continue
        lbp_means[label] = sum_hist / count
        counts[label] = count

    if not lbp_means:
        raise RuntimeError("No LBP histograms were computed. Check image sizes and params.")

    labels_sorted = sorted(lbp_means)

    means_path = args.output_dir / f"lbp_hist_{args.split}.csv"
    with means_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "count"] + [f"bin_{idx}" for idx in range(n_bins)])
        for label in labels_sorted:
            writer.writerow([label, counts[label]] + [f"{value:.6f}" for value in lbp_means[label]])

    distance_matrix = np.zeros((len(labels_sorted), len(labels_sorted)), dtype=np.float64)
    for i, label_i in enumerate(labels_sorted):
        for j, label_j in enumerate(labels_sorted):
            distance_matrix[i, j] = euclidean_distance(lbp_means[label_i], lbp_means[label_j])

    dist_path = args.output_dir / f"lbp_distance_{args.split}.csv"
    with dist_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + labels_sorted)
        for label, row in zip(labels_sorted, distance_matrix, strict=False):
            writer.writerow([label] + [f"{value:.6f}" for value in row])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(distance_matrix, cmap="viridis")
    ax.set_title("LBP mean Euclidean distance")
    ax.set_xticks(range(len(labels_sorted)), labels=labels_sorted, rotation=45, ha="right")
    ax.set_yticks(range(len(labels_sorted)), labels=labels_sorted)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"lbp_distance_{args.split}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    x_vals = np.arange(n_bins)
    for label in labels_sorted:
        ax.plot(x_vals, lbp_means[label], label=label, alpha=0.75)
    ax.set_title("LBP mean histogram per class")
    ax.set_xlabel("LBP bin")
    ax.set_ylabel("Normalized count")
    ax.legend(frameon=False, fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"lbp_overlay_{args.split}.png", dpi=150)
    plt.close(fig)

    print(f"Wrote LBP histograms to {means_path}")
    print(f"Wrote LBP distance matrix to {dist_path}")
    print(f"Wrote LBP overlay plot to {args.output_dir / f'lbp_overlay_{args.split}.png'}")


if __name__ == "__main__":
    main()
