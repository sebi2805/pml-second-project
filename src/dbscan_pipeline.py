from __future__ import annotations

from pathlib import Path
import argparse
from collections import Counter
import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    fbeta_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from cluster_metrics import (
    apply_label_mapping,
    binary_match_counts,
    hungarian_match_counts,
    hungarian_match_mapping,
)
from data_io import discover_samples
from features import build_feature_matrix, build_orb_codebook

DATA_ROOT = Path("data")
SPLIT = "train"
OUTPUT_DIR = Path("output")
BINS = 32
RESIZE = (128, 128)
DEFAULT_FEATURE_SET = "set2"
FEATURE_SETS = ("set1", "set2", "set3", "set4")
HOG_PARAMS = {"orientations": 9, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2)}
LBP_PARAMS = {"radius": 1, "n_points": 8, "method": "uniform"}
BOVW_NUM_WORDS = 64
BOVW_MAX_FEATURES = 500
BOVW_MAX_DESCRIPTORS = 10_000

CLUSTER_PCA_COMPONENTS = 20
PCA_WHITEN = False
POST_PCA_STANDARDIZE = True
K_DISTANCE_PLOT = True
HEALTHY_LABELS = {"Healthy_Nail"}

DEFAULT_DBSCAN = {"eps": 0.6, "min_samples": 5, "metric": "euclidean", "bins": BINS}
GRID_SEARCH = {
    "bins": [16, 32, 64],
    "eps": np.linspace(0.5, 2, 10),
    "min_samples": [3, 5, 8, 12],
    "metric": ["euclidean"],
    "pca_components": [CLUSTER_PCA_COMPONENTS],
}
RUN_GRID_SEARCH = True


def iter_dbscan_configs(feature_set: str) -> list[dict[str, float | int | str | None]]:
    if not RUN_GRID_SEARCH:
        return [{**DEFAULT_DBSCAN, "pca_components": CLUSTER_PCA_COMPONENTS}]

    configs = []
    bins_values = GRID_SEARCH["bins"] if feature_set == "set1" else [BINS]
    for bins in bins_values:
        for eps in GRID_SEARCH["eps"]:
            for min_samples in GRID_SEARCH["min_samples"]:
                for metric in GRID_SEARCH["metric"]:
                    for pca_components in GRID_SEARCH["pca_components"]:
                        configs.append(
                            {
                                "bins": bins,
                                "eps": eps,
                                "min_samples": min_samples,
                                "metric": metric,
                                "pca_components": pca_components,
                            }
                        )
    return configs


def feature_tag(bins: int, feature_set: str) -> str:
    if feature_set == "set1":
        return f"bins{bins}"
    return feature_set


def pca_tag(components: int | None) -> str:
    if components is None:
        return "pcaNone"
    return f"pca{components}"


def build_summary_columns(feature_set: str) -> list[str]:
    columns = [
        "feature_set",
        "resize_w",
        "resize_h",
        "pca_whiten",
        "post_pca_standardize",
    ]
    if feature_set == "set1":
        columns.append("bins")
    elif feature_set == "set2":
        columns.extend(
            [
                "hog_orientations",
                "hog_pixels_per_cell_w",
                "hog_pixels_per_cell_h",
                "hog_cells_per_block_w",
                "hog_cells_per_block_h",
            ]
        )
    elif feature_set == "set3":
        columns.extend(
            [
                "bovw_num_words",
                "bovw_max_features",
                "bovw_max_descriptors",
                "orb_max_features",
            ]
        )
    elif feature_set == "set4":
        columns.extend(["lbp_radius", "lbp_n_points", "lbp_method"])
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    return columns


def build_summary_prefix(feature_set: str, bins: int) -> list[str]:
    if RESIZE is None:
        resize_w = ""
        resize_h = ""
    else:
        resize_w, resize_h = RESIZE

    values = [
        feature_set,
        str(resize_w),
        str(resize_h),
        str(PCA_WHITEN),
        str(POST_PCA_STANDARDIZE),
    ]
    if feature_set == "set1":
        values.append(str(bins))
    elif feature_set == "set2":
        values.extend(
            [
                str(HOG_PARAMS["orientations"]),
                str(HOG_PARAMS["pixels_per_cell"][0]),
                str(HOG_PARAMS["pixels_per_cell"][1]),
                str(HOG_PARAMS["cells_per_block"][0]),
                str(HOG_PARAMS["cells_per_block"][1]),
            ]
        )
    elif feature_set == "set3":
        values.extend(
            [
                str(BOVW_NUM_WORDS),
                str(BOVW_MAX_FEATURES),
                str(BOVW_MAX_DESCRIPTORS),
                str(BOVW_MAX_FEATURES),
            ]
        )
    elif feature_set == "set4":
        values.extend(
            [
                str(LBP_PARAMS["radius"]),
                str(LBP_PARAMS["n_points"]),
                str(LBP_PARAMS["method"]),
            ]
        )
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    return values


def summarize_labels(labels: np.ndarray) -> tuple[int, int]:
    unique, counts = np.unique(labels, return_counts=True)
    cluster_count = int(np.sum(unique != -1))
    noise_count = int(counts[unique == -1][0]) if np.any(unique == -1) else 0
    return cluster_count, noise_count


def plot_clusters(x_2d: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    unique_labels = sorted(set(labels))
    num_labels = len(unique_labels)
    cmap = plt.cm.get_cmap("tab20", max(1, num_labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        color = "black" if label == -1 else cmap(idx)
        name = "noise" if label == -1 else f"cluster {label}"
        ax.scatter(x_2d[mask, 0], x_2d[mask, 1], s=18, alpha=0.75, color=color, label=name)

    ax.set_title("DBSCAN clusters (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    if num_labels <= 15:
        ax.legend(frameon=False, fontsize="small")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_k_distance(features: np.ndarray, k: int, output_path: Path) -> None:
    if len(features) < 2:
        return
    k = max(1, min(k, len(features) - 1))
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(features)
    distances, _ = neighbors.kneighbors(features)
    k_distances = np.sort(distances[:, -1])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(np.arange(1, len(k_distances) + 1), k_distances)
    ax.set_title(f"k-distance plot (k={k})")
    ax.set_xlabel("Samples sorted")
    ax.set_ylabel("Distance to k-th neighbor")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def reduce_for_clustering(features_scaled: np.ndarray, components: int | None) -> np.ndarray:
    if components is None or components <= 0:
        return features_scaled
    if components >= features_scaled.shape[1]:
        return features_scaled
    pca = PCA(
        n_components=components,
        whiten=PCA_WHITEN,
        random_state=42,
    )
    return pca.fit_transform(features_scaled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DBSCAN over image features.")
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default=DEFAULT_FEATURE_SET,
        help="Feature set to use: set1 (color stats), set2 (HOG), set3 (BoVW), set4 (LBP).",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    feature_set = args.feature_set

    samples = discover_samples(DATA_ROOT, SPLIT)
    if not samples:
        raise RuntimeError(f"No samples found under {DATA_ROOT / SPLIT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = OUTPUT_DIR / "dbscan" / feature_set
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

    configs = iter_dbscan_configs(feature_set)
    bins_values = sorted({int(config["bins"]) for config in configs})
    features_cache: dict[tuple[int, int | None], tuple[np.ndarray, np.ndarray, list[str]]] = {}

    def get_features_for_bins(
        bins: int, pca_components: int | None
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        cache_key = (bins, pca_components)
        cached = features_cache.get(cache_key)
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
        elif feature_set == "set4":
            features, labels = build_feature_matrix(
                samples,
                resize=RESIZE,
                feature_set=feature_set,
                lbp_params=LBP_PARAMS,
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
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        features_cluster = reduce_for_clustering(features_scaled, pca_components)
        if POST_PCA_STANDARDIZE and features_cluster.size:
            features_cluster = StandardScaler().fit_transform(features_cluster)

        if features_cluster.shape[1] >= 2:
            x_2d = PCA(n_components=2, random_state=42).fit_transform(features_cluster)
        else:
            x_2d = np.column_stack([features_cluster[:, 0], np.zeros(len(features_cluster))])

        features_cache[cache_key] = (features_cluster, x_2d, labels)
        return features_cluster, x_2d, labels

    if K_DISTANCE_PLOT:
        if RUN_GRID_SEARCH:
            k = int(min(GRID_SEARCH["min_samples"]))
        else:
            k = int(DEFAULT_DBSCAN["min_samples"])
        pca_values = sorted(
            {config.get("pca_components") for config in configs},
            key=lambda value: (value is None, value if value is not None else -1),
        )
        for bins in bins_values:
            for pca_components in pca_values:
                features_cluster, _, _ = get_features_for_bins(bins, pca_components)
                kdist_path = output_dir / (
                    f"dbscan_{SPLIT}_{feature_tag(bins, feature_set)}_{pca_tag(pca_components)}_kdistance_k{k}.png"
                )
                plot_k_distance(features_cluster, k, kdist_path)

    if feature_set == "set1":
        results_path = output_dir / f"dbscan_{SPLIT}_summary.csv"
    else:
        results_path = output_dir / f"dbscan_{SPLIT}_{feature_set}_summary.csv"
    summary_columns = build_summary_columns(feature_set)
    metric_columns = [
        "pca_components",
        "eps",
        "min_samples",
        "metric",
        "clusters",
        "noise",
        "noise_ratio",
        "accuracy_clustered",
        "coverage",
        "accuracy_overall",
        "accuracy_binary_clustered",
        "accuracy_binary_overall",
        "precision_clustered",
        "recall_clustered",
        "f2_clustered",
        "ari_clustered",
        "nmi_clustered",
        "precision_overall",
        "recall_overall",
        "f2_overall",
        "ari_overall",
        "nmi_overall",
    ]
    rows = [",".join(summary_columns + metric_columns) + "\n"]

    for config in configs:
        bins = int(config["bins"])
        pca_components = config.get("pca_components", CLUSTER_PCA_COMPONENTS)
        features_cluster, x_2d, labels = get_features_for_bins(bins, pca_components)
        dbscan = DBSCAN(
            eps=float(config["eps"]),
            min_samples=int(config["min_samples"]),
            metric=str(config["metric"]),
        )
        cluster_labels = dbscan.fit_predict(features_cluster)

        eps = config["eps"]
        min_samples = config["min_samples"]
        metric = config["metric"]
        output_path = output_dir / (
            f"dbscan_{SPLIT}_{feature_tag(bins, feature_set)}_{pca_tag(pca_components)}_eps{eps}_min{min_samples}_{metric}_pca.png"
        )
        plot_clusters(x_2d, cluster_labels, output_path)

        cluster_count, noise_count = summarize_labels(cluster_labels)
        total_samples = len(cluster_labels)
        noise_ratio = noise_count / total_samples if total_samples else 0.0
        clustered_mask = cluster_labels != -1
        clustered_count = int(np.sum(clustered_mask))
        coverage = clustered_count / total_samples if total_samples else 0.0
        labels_array = np.asarray(labels)
        class_labels = sorted(set(labels))
        if clustered_count:
            clustered_true = labels_array[clustered_mask]
            clustered_pred = cluster_labels[clustered_mask]
            correct_clustered, _ = hungarian_match_counts(clustered_true, clustered_pred)
            accuracy_clustered = correct_clustered / clustered_count
            accuracy_overall = correct_clustered / total_samples if total_samples else 0.0
            correct_binary, _ = binary_match_counts(clustered_true, clustered_pred, HEALTHY_LABELS)
            accuracy_binary_clustered = correct_binary / clustered_count
            accuracy_binary_overall = correct_binary / total_samples if total_samples else 0.0
            mapping = hungarian_match_mapping(clustered_true, clustered_pred)
            mapped_clustered = apply_label_mapping(clustered_pred, mapping, fallback_label="__noise__")
            mapped_all = apply_label_mapping(cluster_labels, mapping, fallback_label="__noise__")
            precision_clustered, recall_clustered, _, _ = precision_recall_fscore_support(
                clustered_true,
                mapped_clustered,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
            f2_clustered = fbeta_score(
                clustered_true,
                mapped_clustered,
                labels=class_labels,
                beta=2,
                average="macro",
                zero_division=0,
            )
            precision_overall, recall_overall, _, _ = precision_recall_fscore_support(
                labels_array,
                mapped_all,
                labels=class_labels,
                average="macro",
                zero_division=0,
            )
            f2_overall = fbeta_score(
                labels_array,
                mapped_all,
                labels=class_labels,
                beta=2,
                average="macro",
                zero_division=0,
            )
            ari_clustered = adjusted_rand_score(clustered_true, clustered_pred)
            nmi_clustered = normalized_mutual_info_score(clustered_true, clustered_pred)
            ari_overall = adjusted_rand_score(labels_array, cluster_labels)
            nmi_overall = normalized_mutual_info_score(labels_array, cluster_labels)
        else:
            accuracy_clustered = 0.0
            accuracy_overall = 0.0
            accuracy_binary_clustered = 0.0
            accuracy_binary_overall = 0.0
            precision_clustered = 0.0
            recall_clustered = 0.0
            f2_clustered = 0.0
            precision_overall = 0.0
            recall_overall = 0.0
            f2_overall = 0.0
            ari_clustered = 0.0
            nmi_clustered = 0.0
            ari_overall = 0.0
            nmi_overall = 0.0
        rows.append(
            ",".join(
                build_summary_prefix(feature_set, bins)
                + [
                    str(pca_components),
                    str(eps),
                    str(min_samples),
                    str(metric),
                    str(cluster_count),
                    str(noise_count),
                    f"{noise_ratio:.4f}",
                    f"{accuracy_clustered:.4f}",
                    f"{coverage:.4f}",
                    f"{accuracy_overall:.4f}",
                    f"{accuracy_binary_clustered:.4f}",
                    f"{accuracy_binary_overall:.4f}",
                    f"{precision_clustered:.4f}",
                    f"{recall_clustered:.4f}",
                    f"{f2_clustered:.4f}",
                    f"{ari_clustered:.4f}",
                    f"{nmi_clustered:.4f}",
                    f"{precision_overall:.4f}",
                    f"{recall_overall:.4f}",
                    f"{f2_overall:.4f}",
                    f"{ari_overall:.4f}",
                    f"{nmi_overall:.4f}",
                ]
            )
            + "\n"
        )
        print(f"Saved plot to {output_path}")

        report_path = output_dir / (
            f"dbscan_{SPLIT}_{feature_tag(bins, feature_set)}_{pca_tag(pca_components)}_eps{eps}_min{min_samples}_{metric}_clusters.csv"
        )
        report_rows = build_cluster_report(labels, cluster_labels)
        report_path.write_text("".join(report_rows), encoding="utf-8")
        print(f"Saved cluster report to {report_path}")

    results_path.write_text("".join(rows), encoding="utf-8")
    print(f"Saved summary to {results_path}")


if __name__ == "__main__":
    main()
