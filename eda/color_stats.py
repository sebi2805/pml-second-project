from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.data_io import discover_samples, load_image

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUTPUT_DIR = Path("output") / "eda"
DEFAULT_SPLIT = "train"
DEFAULT_RESIZE = (128, 128)
DEFAULT_BINS = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute mean images and HSV histograms per class."
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
        help="Resize images for mean/hist computations.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of bins for S/V histograms (H uses half this).",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=0,
        help="Optional cap per class (0 means no cap).",
    )
    return parser.parse_args()


def safe_name(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in label)


def plot_hsv_histogram(hist_h: np.ndarray, hist_s: np.ndarray, hist_v: np.ndarray, output_path: Path) -> None:
    if hist_h.sum() > 0:
        hist_h = hist_h / hist_h.sum()
    if hist_s.sum() > 0:
        hist_s = hist_s / hist_s.sum()
    if hist_v.sum() > 0:
        hist_v = hist_v / hist_v.sum()

    h_bins = np.arange(hist_h.size)
    sv_bins = np.arange(hist_s.size)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h_bins, hist_h, label="H", color="#1f77b4")
    ax.plot(sv_bins, hist_s, label="S", color="#2ca02c")
    ax.plot(sv_bins, hist_v, label="V", color="#ff7f0e")
    ax.set_title("HSV histogram")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Normalized count")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def main() -> None:
    args = parse_args()
    samples = discover_samples(args.data_root, args.split)
    if not samples:
        raise RuntimeError(f"No samples found under {args.data_root / args.split}")

    resize = tuple(args.resize)
    h_bins = max(1, args.bins // 2)
    sv_bins = max(1, args.bins)

    per_class_samples: dict[str, list[Path]] = defaultdict(list)
    for sample in samples:
        per_class_samples[sample.label].append(sample.path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mean_dir = args.output_dir / "mean_images"
    mean_dir.mkdir(parents=True, exist_ok=True)

    hsv_means_rows = [["label", "count", "mean_h", "mean_s", "mean_v"]]
    h_histograms: dict[str, np.ndarray] = {}

    overall_sum = None
    overall_count = 0
    overall_hist_h = np.zeros(h_bins, dtype=np.float64)
    overall_hist_s = np.zeros(sv_bins, dtype=np.float64)
    overall_hist_v = np.zeros(sv_bins, dtype=np.float64)
    overall_hsv_sum = np.zeros(3, dtype=np.float64)
    overall_pixels = 0

    for label in sorted(per_class_samples):
        paths = per_class_samples[label]
        if args.max_images_per_class > 0:
            paths = paths[: args.max_images_per_class]

        if not paths:
            continue

        sum_image = None
        hist_h = np.zeros(h_bins, dtype=np.float64)
        hist_s = np.zeros(sv_bins, dtype=np.float64)
        hist_v = np.zeros(sv_bins, dtype=np.float64)
        hsv_sum = np.zeros(3, dtype=np.float64)
        pixel_count = 0

        for path in paths:
            image = load_image(path, resize=resize)
            if sum_image is None:
                sum_image = np.zeros_like(image, dtype=np.float64)
            sum_image += image

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            pixels = hsv.reshape(-1, 3)
            pixel_count += pixels.shape[0]
            hsv_sum += pixels.sum(axis=0)

            hist_h += np.histogram(pixels[:, 0], bins=h_bins, range=(0, 180))[0]
            hist_s += np.histogram(pixels[:, 1], bins=sv_bins, range=(0, 256))[0]
            hist_v += np.histogram(pixels[:, 2], bins=sv_bins, range=(0, 256))[0]

        mean_image = (sum_image / len(paths)).round().astype(np.uint8)
        label_safe = safe_name(label)
        mean_path = mean_dir / f"{args.split}_{label_safe}_mean.png"
        cv2.imwrite(str(mean_path), mean_image)

        plot_hsv_histogram(
            hist_h,
            hist_s,
            hist_v,
            args.output_dir / f"hsv_hist_{args.split}_{label_safe}.png",
        )

        hist_h_norm = hist_h / max(1.0, hist_h.sum())
        h_histograms[label] = hist_h_norm

        mean_hsv = hsv_sum / max(1, pixel_count)
        hsv_means_rows.append(
            [label, len(paths), f"{mean_hsv[0]:.2f}", f"{mean_hsv[1]:.2f}", f"{mean_hsv[2]:.2f}"]
        )

        if overall_sum is None:
            overall_sum = np.zeros_like(mean_image, dtype=np.float64)
        overall_sum += sum_image
        overall_count += len(paths)
        overall_hist_h += hist_h
        overall_hist_s += hist_s
        overall_hist_v += hist_v
        overall_hsv_sum += hsv_sum
        overall_pixels += pixel_count

    if overall_sum is not None and overall_count > 0:
        overall_mean = (overall_sum / overall_count).round().astype(np.uint8)
        cv2.imwrite(str(mean_dir / f"{args.split}_overall_mean.png"), overall_mean)
        plot_hsv_histogram(
            overall_hist_h,
            overall_hist_s,
            overall_hist_v,
            args.output_dir / f"hsv_hist_{args.split}_overall.png",
        )
        overall_hsv = overall_hsv_sum / max(1, overall_pixels)
        hsv_means_rows.append(
            [
                "overall",
                overall_count,
                f"{overall_hsv[0]:.2f}",
                f"{overall_hsv[1]:.2f}",
                f"{overall_hsv[2]:.2f}",
            ]
        )

    hsv_means_path = args.output_dir / f"hsv_means_{args.split}.csv"
    with hsv_means_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(hsv_means_rows)

    if h_histograms:
        labels_sorted = sorted(h_histograms)
        hist_rows = [["label"] + [f"h_bin_{idx}" for idx in range(h_bins)]]
        for label in labels_sorted:
            hist_rows.append([label] + [f"{value:.6f}" for value in h_histograms[label]])
        h_hist_path = args.output_dir / f"h_hist_{args.split}.csv"
        with h_hist_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(hist_rows)

        distance_matrix = np.zeros((len(labels_sorted), len(labels_sorted)), dtype=np.float64)
        for i, label_i in enumerate(labels_sorted):
            for j, label_j in enumerate(labels_sorted):
                distance_matrix[i, j] = euclidean_distance(
                    h_histograms[label_i], h_histograms[label_j]
                )

        dist_path = args.output_dir / f"h_hist_distance_{args.split}.csv"
        dist_rows = [["label"] + labels_sorted]
        for label, row in zip(labels_sorted, distance_matrix, strict=False):
            dist_rows.append([label] + [f"{value:.6f}" for value in row])
        with dist_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(dist_rows)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(distance_matrix, cmap="viridis")
        ax.set_title("H histogram Euclidean distance")
        ax.set_xticks(range(len(labels_sorted)), labels=labels_sorted, rotation=45, ha="right")
        ax.set_yticks(range(len(labels_sorted)), labels=labels_sorted)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"h_hist_distance_{args.split}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        x_bins = np.arange(h_bins)
        for label in labels_sorted:
            ax.plot(x_bins, h_histograms[label], label=label, alpha=0.75)
        ax.set_title("H histogram per class")
        ax.set_xlabel("Hue bin")
        ax.set_ylabel("Normalized count")
        ax.legend(frameon=False, fontsize="small", ncol=2)
        fig.tight_layout()
        fig.savefig(args.output_dir / f"h_hist_overlay_{args.split}.png", dpi=150)
        plt.close(fig)

    print(f"Wrote mean images to {mean_dir}")
    print(f"Wrote HSV histograms to {args.output_dir}")
    print(f"Wrote HSV means to {hsv_means_path}")
    if h_histograms:
        print(f"Wrote H-only histograms to {h_hist_path}")
        print(f"Wrote H-only distance matrix to {dist_path}")


if __name__ == "__main__":
    main()
