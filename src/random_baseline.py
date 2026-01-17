from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

from data_io import discover_samples

DATA_ROOT = Path("data")
DEFAULT_SPLIT = "validation"
DEFAULT_RUNS = 50
DEFAULT_SEED = 42
K_MIN = 2
K_MAX = 10
OUTPUT_CSV: Optional[Path] = None


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    total = y_true.size
    purity_total = 0
    for cluster_id in np.unique(y_pred):
        mask = y_pred == cluster_id
        if not np.any(mask):
            continue
        _, counts = np.unique(y_true[mask], return_counts=True)
        purity_total += int(counts.max())
    return purity_total / total


def mean_std(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    if values.size == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def run_random_baseline(
    split: str = DEFAULT_SPLIT,
    runs: int = DEFAULT_RUNS,
    seed: int = DEFAULT_SEED,
    k_values: Iterable[int] = range(K_MIN, K_MAX + 1),
    output_csv: Optional[Path] = OUTPUT_CSV,
) -> None:
    samples = discover_samples(DATA_ROOT, split)
    if not samples:
        raise RuntimeError(f"No samples found under {DATA_ROOT / split}")
    if runs <= 0:
        raise ValueError("runs must be a positive integer.")

    labels = [sample.label for sample in samples]
    encoder = LabelEncoder()
    y_true = encoder.fit_transform(labels)
    num_classes = len(encoder.classes_)

    results = []
    for k in k_values:
        if k <= 0:
            raise ValueError("k_values must contain positive integers.")
        rng = np.random.default_rng(seed + int(k))
        ari_values = []
        nmi_values = []
        purity_values = []
        for _ in range(runs):
            y_pred = rng.integers(0, k, size=y_true.size)
            ari_values.append(adjusted_rand_score(y_true, y_pred))
            nmi_values.append(normalized_mutual_info_score(y_true, y_pred))
            purity_values.append(purity_score(y_true, y_pred))

        ari_arr = np.array(ari_values, dtype=np.float64)
        nmi_arr = np.array(nmi_values, dtype=np.float64)
        purity_arr = np.array(purity_values, dtype=np.float64)

        ari_mean, ari_std = mean_std(ari_arr)
        nmi_mean, nmi_std = mean_std(nmi_arr)
        purity_mean, purity_std = mean_std(purity_arr)
        results.append((k, ari_mean, ari_std, nmi_mean, nmi_std, purity_mean, purity_std))

    print(
        "Random baseline "
        f"split={split} runs={runs} seed={seed} "
        f"samples={len(y_true)} classes={num_classes}"
    )
    print("k,ari_mean,ari_std,nmi_mean,nmi_std,purity_mean,purity_std")
    for k, ari_mean, ari_std, nmi_mean, nmi_std, purity_mean, purity_std in results:
        print(
            f"{k},{ari_mean:.4f},{ari_std:.4f},"
            f"{nmi_mean:.4f},{nmi_std:.4f},"
            f"{purity_mean:.4f},{purity_std:.4f}"
        )

    if output_csv is not None:
        if output_csv.parent:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            "split,k,runs,seed,samples,classes,ari_mean,ari_std,nmi_mean,nmi_std,purity_mean,purity_std\n"
        ]
        for k, ari_mean, ari_std, nmi_mean, nmi_std, purity_mean, purity_std in results:
            rows.append(
                f"{split},{k},{runs},{seed},{len(y_true)},{num_classes},"
                f"{ari_mean:.6f},{ari_std:.6f},{nmi_mean:.6f},{nmi_std:.6f},"
                f"{purity_mean:.6f},{purity_std:.6f}\n"
            )
        output_csv.write_text("".join(rows), encoding="utf-8")
        print(f"Saved random baseline metrics to {output_csv}")


if __name__ == "__main__":
    run_random_baseline()
