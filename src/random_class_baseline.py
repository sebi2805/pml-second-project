from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from data_io import discover_samples

DATA_ROOT = Path("data")
DEFAULT_SPLIT = "validation"
DEFAULT_RUNS = 50
DEFAULT_SEED = 42
HEALTHY_LABELS = {"Healthy_Nail"}


def mean_std(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    if values.size == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def per_class_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, classes: Iterable[str]
) -> dict[str, float]:
    accuracies: dict[str, float] = {}
    for cls in classes:
        mask = y_true == cls
        if not np.any(mask):
            accuracies[str(cls)] = 0.0
        else:
            accuracies[str(cls)] = float(np.mean(y_pred[mask] == cls))
    return accuracies


def binary_accuracies(
    y_true: np.ndarray, y_pred: np.ndarray, healthy_labels: set[str]
) -> tuple[float, float, float]:
    if y_true.size == 0:
        return 0.0, 0.0, 0.0

    healthy_list = list(healthy_labels)
    true_healthy = np.isin(y_true, healthy_list)
    pred_healthy = np.isin(y_pred, healthy_list)

    binary_acc = float(np.mean(true_healthy == pred_healthy))

    if np.any(true_healthy):
        healthy_acc = float(np.mean(pred_healthy[true_healthy]))
    else:
        healthy_acc = 0.0

    if np.any(~true_healthy):
        unhealthy_acc = float(np.mean(~pred_healthy[~true_healthy]))
    else:
        unhealthy_acc = 0.0

    return binary_acc, healthy_acc, unhealthy_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random class guessing baseline on the validation split."
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to use (default: validation).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Number of random runs to average.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base RNG seed.",
    )
    return parser.parse_args()


def run_random_class_baseline(split: str, runs: int, seed: int) -> None:
    samples = discover_samples(DATA_ROOT, split)
    if not samples:
        raise RuntimeError(f"No samples found under {DATA_ROOT / split}")
    if runs <= 0:
        raise ValueError("runs must be a positive integer.")

    labels = np.array([sample.label for sample in samples], dtype=str)
    classes = sorted({str(label) for label in labels})

    class_counts = {cls: int(np.sum(labels == cls)) for cls in classes}
    healthy_count = int(np.sum(np.isin(labels, list(HEALTHY_LABELS))))
    unhealthy_count = int(labels.size - healthy_count)

    overall_accs: list[float] = []
    binary_accs: list[float] = []
    healthy_accs: list[float] = []
    unhealthy_accs: list[float] = []
    per_class_accs = {cls: [] for cls in classes}

    for run in range(runs):
        rng = np.random.default_rng(seed + run)
        y_pred = rng.choice(classes, size=labels.size, replace=True)

        overall_accs.append(float(np.mean(y_pred == labels)))

        per_class = per_class_accuracy(labels, y_pred, classes)
        for cls, acc in per_class.items():
            per_class_accs[cls].append(acc)

        binary_acc, healthy_acc, unhealthy_acc = binary_accuracies(
            labels, y_pred, HEALTHY_LABELS
        )
        binary_accs.append(binary_acc)
        healthy_accs.append(healthy_acc)
        unhealthy_accs.append(unhealthy_acc)

    overall_mean, overall_std = mean_std(np.array(overall_accs, dtype=np.float64))
    binary_mean, binary_std = mean_std(np.array(binary_accs, dtype=np.float64))
    healthy_mean, healthy_std = mean_std(np.array(healthy_accs, dtype=np.float64))
    unhealthy_mean, unhealthy_std = mean_std(np.array(unhealthy_accs, dtype=np.float64))

    print(
        "Random class baseline "
        f"split={split} runs={runs} seed={seed} "
        f"samples={labels.size} classes={len(classes)}"
    )
    print(f"overall_accuracy_mean,overall_accuracy_std={overall_mean:.4f},{overall_std:.4f}")
    print(f"binary_accuracy_mean,binary_accuracy_std={binary_mean:.4f},{binary_std:.4f}")
    print(f"healthy_accuracy_mean,healthy_accuracy_std={healthy_mean:.4f},{healthy_std:.4f}")
    print(
        f"unhealthy_accuracy_mean,unhealthy_accuracy_std={unhealthy_mean:.4f},{unhealthy_std:.4f}"
    )
    print(f"healthy_count={healthy_count} unhealthy_count={unhealthy_count}")
    print("class,count,accuracy_mean,accuracy_std")
    for cls in classes:
        acc_mean, acc_std = mean_std(np.array(per_class_accs[cls], dtype=np.float64))
        print(f"{cls},{class_counts[cls]},{acc_mean:.4f},{acc_std:.4f}")


def main() -> None:
    args = parse_args()
    run_random_class_baseline(args.split, args.runs, args.seed)


if __name__ == "__main__":
    main()
