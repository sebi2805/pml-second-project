import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_io import discover_samples, load_image
from src.features import extract_feature_set_2

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUTPUT_DIR = Path("output") / "eda"
DEFAULT_SPLIT = "train"
DEFAULT_RESIZE = (128, 128)
DEFAULT_ORIENTATIONS = 9
DEFAULT_PIXELS_PER_CELL = (8, 8)
DEFAULT_CELLS_PER_BLOCK = (2, 2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute mean HOG descriptors per class and compare them."
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
        help="Resize images for HOG extraction.",
    )
    parser.add_argument(
        "--orientations",
        type=int,
        default=DEFAULT_ORIENTATIONS,
        help="Number of HOG orientation bins.",
    )
    parser.add_argument(
        "--pixels-per-cell",
        type=int,
        nargs=2,
        default=list(DEFAULT_PIXELS_PER_CELL),
        metavar=("WIDTH", "HEIGHT"),
        help="Pixels per HOG cell.",
    )
    parser.add_argument(
        "--cells-per-block",
        type=int,
        nargs=2,
        default=list(DEFAULT_CELLS_PER_BLOCK),
        metavar=("WIDTH", "HEIGHT"),
        help="Cells per HOG block.",
    )
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        default=0,
        help="Optional cap per class (0 means no cap).",
    )
    parser.add_argument(
        "--plot-step",
        type=int,
        default=1,
        help="Stride for plotting descriptor dimensions (1 uses all points).",
    )
    return parser.parse_args()


def euclidean_distance(a, b):
    return float(np.linalg.norm(a - b))


def main():
    args = parse_args()
    samples = discover_samples(args.data_root, args.split)
    if not samples:
        print(f"No samples found under {args.data_root / args.split}")
        return

    resize = tuple(args.resize)
    pixels_per_cell = tuple(args.pixels_per_cell)
    cells_per_block = tuple(args.cells_per_block)

    per_class_samples = defaultdict(list)
    for sample in samples:
        per_class_samples[sample.label].append(sample.path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    hog_means = {}
    skipped = 0
    for label in sorted(per_class_samples):
        paths = per_class_samples[label]
        if args.max_images_per_class > 0:
            paths = paths[: args.max_images_per_class]

        sum_desc = None
        count = 0
        for path in paths:
            image = load_image(path, resize=resize)
            if image is None:
                print(f"Skipping image {path}")
                skipped += 1
                continue
            desc = extract_feature_set_2(
                image,
                resize=None,
                orientations=args.orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
            )
            if desc.size == 0:
                skipped += 1
                continue
            if sum_desc is None:
                sum_desc = np.zeros_like(desc, dtype=np.float64)
            elif sum_desc.shape != desc.shape:
                print(
                    f"HOG descriptor size mismatch for label {label} "
                    f"expected {sum_desc.shape} got {desc.shape}"
                )
                skipped += 1
                continue
            sum_desc += desc
            count += 1

        if sum_desc is None or count == 0:
            continue
        hog_means[label] = (sum_desc / count).astype(np.float64)

    if not hog_means:
        print("No HOG descriptors were computed check image sizes and params")
        return

    labels_sorted = sorted(hog_means)
    dim = int(next(iter(hog_means.values())).shape[0])

    means_path = args.output_dir / f"hog_means_{args.split}.csv"
    with means_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + [f"dim_{idx}" for idx in range(dim)])
        for label in labels_sorted:
            writer.writerow([label] + [f"{value:.6f}" for value in hog_means[label]])

    distance_matrix = np.zeros((len(labels_sorted), len(labels_sorted)), dtype=np.float64)
    for i, label_i in enumerate(labels_sorted):
        for j, label_j in enumerate(labels_sorted):
            distance_matrix[i, j] = euclidean_distance(hog_means[label_i], hog_means[label_j])

    dist_path = args.output_dir / f"hog_distance_{args.split}.csv"
    with dist_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"] + labels_sorted)
        for label, row in zip(labels_sorted, distance_matrix, strict=False):
            writer.writerow([label] + [f"{value:.6f}" for value in row])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(distance_matrix, cmap="viridis")
    ax.set_title("HOG mean Euclidean distance")
    ax.set_xticks(range(len(labels_sorted)), labels=labels_sorted, rotation=45, ha="right")
    ax.set_yticks(range(len(labels_sorted)), labels=labels_sorted)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"hog_distance_{args.split}.png", dpi=150)
    plt.close(fig)

    plot_step = max(1, args.plot_step)
    x_vals = np.arange(dim)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label in labels_sorted:
        ax.plot(
            x_vals[::plot_step],
            hog_means[label][::plot_step],
            label=label,
            alpha=0.75,
        )
    ax.set_title("HOG mean descriptor per class")
    ax.set_xlabel("Descriptor index")
    ax.set_ylabel("Value")
    ax.legend(frameon=False, fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"hog_overlay_{args.split}.png", dpi=150)
    plt.close(fig)

    print(f"Wrote HOG means to {means_path}")
    print(f"Wrote HOG distance matrix to {dist_path}")
    print(f"Wrote HOG overlay plot to {args.output_dir / f'hog_overlay_{args.split}.png'}")
    if skipped:
        print(f"Skipped {skipped} images with empty HOG descriptors")


if __name__ == "__main__":
    main()
