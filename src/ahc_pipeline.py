from __future__ import annotations

from pathlib import Path
import argparse
from collections import Counter
import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_io import discover_samples
from features import build_feature_matrix, build_orb_codebook

DATA_ROOT = Path("data")
SPLIT = "train"
OUTPUT_DIR = Path("output")
BINS = 32
RESIZE = (128, 128)
DEFAULT_FEATURE_SET = "set2"
FEATURE_SETS = ("set1", "set2", "set3")
HOG_PARAMS = {"orientations": 9, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2)}
BOVW_NUM_WORDS = 64
BOVW_MAX_FEATURES = 500
BOVW_MAX_DESCRIPTORS = 10_000

DEFAULT_AHC = {
    "n_clusters": 6,
    "linkage": "ward",
    "metric": "euclidean",
    "bins": BINS,
}
GRID_SEARCH = {
    "bins": [16, 32, 64],
    "n_clusters": [4, 6, 8, 10],
    "linkage": ["ward", "average", "complete"],
    "metric": ["euclidean", "cosine"],
}
RUN_GRID_SEARCH = True

CLUSTER_PCA_COMPONENTS = 20
PCA_WHITEN = False
POST_PCA_STANDARDIZE = True


def iter_ahc_configs(feature_set: str) -> list[dict[str, int | str]]:
    if not RUN_GRID_SEARCH:
        return [DEFAULT_AHC]

    configs: list[dict[str, int | str]] = []
    bins_values = GRID_SEARCH["bins"] if feature_set == "set1" else [BINS]
    for bins in bins_values:
        for n_clusters in GRID_SEARCH["n_clusters"]:
            for linkage in GRID_SEARCH["linkage"]:
                for metric in GRID_SEARCH["metric"]:
                    if linkage == "ward" and metric != "euclidean":
                        continue
                    configs.append(
                        {
                            "bins": bins,
                            "n_clusters": n_clusters,
                            "linkage": linkage,
                            "metric": metric,
                        }
                    )
    return configs


def feature_tag(bins: int, feature_set: str) -> str:
    if feature_set == "set1":
        return f"bins{bins}"
    return feature_set


def cluster_entropy(class_counts: Counter[str]) -> float:
    total = sum(class_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in class_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def build_cluster_report(
    labels_true: list[str], cluster_labels: np.ndarray
) -> list[str]:
    clusters: dict[int, Counter[str]] = {}
    for true_label, cluster_label in zip(labels_true, cluster_labels, strict=False):
        clusters.setdefault(int(cluster_label), Counter())[true_label] += 1

    rows = [
        "cluster,size,entropy,purity,top1_label,top1_count,top2_label,top2_count,top3_label,top3_count\n"
    ]
    for cluster_label in sorted(clusters):
        counts = clusters[cluster_label]
        size = sum(counts.values())
        entropy = cluster_entropy(counts)
        if size:
            top = counts.most_common(3)
            purity = top[0][1] / size
        else:
            top = []
            purity = 0.0
        top_entries = top + [("", 0)] * (3 - len(top))
        (t1, c1), (t2, c2), (t3, c3) = top_entries[:3]
        rows.append(
            f"{cluster_label},{size},{entropy:.4f},{purity:.4f},{t1},{c1},{t2},{c2},{t3},{c3}\n"
        )
    return rows


def plot_clusters(x_2d: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    unique_labels = sorted(set(labels))
    num_labels = len(unique_labels)
    cmap = plt.cm.get_cmap("tab20", max(1, num_labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        name = f"cluster {label}"
        ax.scatter(
            x_2d[mask, 0],
            x_2d[mask, 1],
            s=18,
            alpha=0.75,
            color=cmap(idx),
            label=name,
        )

    ax.set_title("AHC clusters (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    if num_labels <= 15:
        ax.legend(frameon=False, fontsize="small")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def reduce_for_clustering(features_scaled: np.ndarray) -> np.ndarray:
    if CLUSTER_PCA_COMPONENTS is None:
        return features_scaled
    if CLUSTER_PCA_COMPONENTS >= features_scaled.shape[1]:
        return features_scaled
    pca = PCA(
        n_components=CLUSTER_PCA_COMPONENTS,
        whiten=PCA_WHITEN,
        random_state=42,
    )
    return pca.fit_transform(features_scaled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Agglomerative Clustering over image features.")
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default=DEFAULT_FEATURE_SET,
        help="Feature set to use: set1 (color stats), set2 (HOG), set3 (BoVW).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_set = args.feature_set

    samples = discover_samples(DATA_ROOT, SPLIT)
    if not samples:
        raise RuntimeError(f"No samples found under {DATA_ROOT / SPLIT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = OUTPUT_DIR / "ahc" / feature_set
    output_dir.mkdir(parents=True, exist_ok=True)

    codebook = None
    if feature_set == "set3":
        codebook = build_orb_codebook(
            samples,
            num_words=BOVW_NUM_WORDS,
            max_features=BOVW_MAX_FEATURES,
            max_descriptors=BOVW_MAX_DESCRIPTORS,
            resize=RESIZE,
        )

    configs = iter_ahc_configs(feature_set)
    bins_values = sorted({int(config["bins"]) for config in configs})
    features_cache: dict[int, tuple[np.ndarray, np.ndarray, list[str]]] = {}

    def get_features_for_bins(bins: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
        cached = features_cache.get(bins)
        if cached is not None:
            return cached

        if feature_set == "set1":
            features, labels = build_feature_matrix(
                samples, bins=bins, resize=RESIZE, feature_set=feature_set
            )
        elif feature_set == "set2":
            features, labels = build_feature_matrix(
                samples,
                resize=RESIZE,
                feature_set=feature_set,
                hog_params=HOG_PARAMS,
            )
        elif feature_set == "set3":
            if codebook is None:
                raise RuntimeError("Codebook not initialized for feature_set='set3'.")
            features, labels = build_feature_matrix(
                samples,
                resize=RESIZE,
                feature_set=feature_set,
                codebook=codebook,
                orb_params={"max_features": BOVW_MAX_FEATURES},
            )
        else:
            raise ValueError(f"Unknown FEATURE_SET: {feature_set}")

        if features.size == 0:
            raise RuntimeError("Feature extraction returned an empty matrix.")

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        features_cluster = reduce_for_clustering(features_scaled)
        if POST_PCA_STANDARDIZE and features_cluster.size:
            features_cluster = StandardScaler().fit_transform(features_cluster)

        if features_cluster.shape[1] >= 2:
            x_2d = PCA(n_components=2, random_state=42).fit_transform(features_cluster)
        else:
            x_2d = np.column_stack([features_cluster[:, 0], np.zeros(len(features_cluster))])

        features_cache[bins] = (features_cluster, x_2d, labels)
        return features_cluster, x_2d, labels

    if feature_set == "set1":
        results_path = output_dir / f"ahc_{SPLIT}_summary.csv"
    else:
        results_path = output_dir / f"ahc_{SPLIT}_{feature_set}_summary.csv"
    rows = ["bins,n_clusters,linkage,metric,clusters\n"]

    for config in configs:
        bins = int(config["bins"])
        n_clusters = int(config["n_clusters"])
        linkage = str(config["linkage"])
        metric = str(config["metric"])

        features_cluster, x_2d, labels = get_features_for_bins(bins)
        if n_clusters > len(features_cluster):
            print(
                f"Skipping n_clusters={n_clusters} because only {len(features_cluster)} samples are available."
            )
            continue

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
        )
        cluster_labels = model.fit_predict(features_cluster)

        output_path = output_dir / (
            f"ahc_{SPLIT}_{feature_tag(bins, feature_set)}_n{n_clusters}_link{linkage}_{metric}_pca.png"
        )
        plot_clusters(x_2d, cluster_labels, output_path)

        cluster_count = len(np.unique(cluster_labels))
        rows.append(
            f"{bins},{n_clusters},{linkage},{metric},{cluster_count}\n"
        )
        print(f"Saved plot to {output_path}")

        report_path = output_dir / (
            f"ahc_{SPLIT}_{feature_tag(bins, feature_set)}_n{n_clusters}_link{linkage}_{metric}_clusters.csv"
        )
        report_rows = build_cluster_report(labels, cluster_labels)
        report_path.write_text("".join(report_rows), encoding="utf-8")
        print(f"Saved cluster report to {report_path}")

    results_path.write_text("".join(rows), encoding="utf-8")
    print(f"Saved summary to {results_path}")


if __name__ == "__main__":
    main()
