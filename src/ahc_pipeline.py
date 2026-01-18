from __future__ import annotations

from pathlib import Path
import argparse
from collections import Counter
import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    fbeta_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

from cluster_metrics import (
    binary_accuracy,
    hungarian_match_accuracy,
    hungarian_match_predict,
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
HEALTHY_LABELS = {"Healthy_Nail"}

DEFAULT_AHC = {
    "n_clusters": 6,
    "linkage": "ward",
    "metric": "euclidean",
    "bins": BINS,
    "pca_components": CLUSTER_PCA_COMPONENTS,
}
GRID_SEARCH = {
    "bins": [16, 32, 64],
    "n_clusters": [4, 6, 8, 10],
    "linkage": ["ward", "average", "complete"],
    "metric": ["euclidean", "cosine"],
    "pca_components": [CLUSTER_PCA_COMPONENTS],
}
RUN_GRID_SEARCH = True
PLOT_HEATMAP = True
PLOT_CLUSTERED_HEATMAP = True
HEATMAP_MAX_TICKS = 25


def iter_ahc_configs(feature_set: str) -> list[dict[str, int | str | None]]:
    if not RUN_GRID_SEARCH:
        return [DEFAULT_AHC]

    configs: list[dict[str, int | str | None]] = []
    bins_values = GRID_SEARCH["bins"] if feature_set == "set1" else [BINS]
    for bins in bins_values:
        for n_clusters in GRID_SEARCH["n_clusters"]:
            for linkage in GRID_SEARCH["linkage"]:
                for metric in GRID_SEARCH["metric"]:
                    if linkage == "ward" and metric != "euclidean":
                        continue
                    for pca_components in GRID_SEARCH["pca_components"]:
                        configs.append(
                            {
                                "bins": bins,
                                "n_clusters": n_clusters,
                                "linkage": linkage,
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


def _select_tick_positions(count: int, max_ticks: int) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=int)
    if count <= max_ticks:
        return np.arange(count)
    return np.linspace(0, count - 1, num=max_ticks, dtype=int)


def plot_feature_heatmap(
    features: np.ndarray,
    sample_names: list[str],
    output_path: Path,
    title: str,
    feature_names: list[str] | None = None,
    max_ticks: int = HEATMAP_MAX_TICKS,
    show_sample_labels: bool = False,
    cluster_labels: np.ndarray | None = None,
) -> None:
    if features.size == 0:
        return

    if cluster_labels is not None:
        order = np.argsort(cluster_labels)
        features = features[order]
        sample_names = [sample_names[idx] for idx in order]
        sorted_labels = np.asarray(cluster_labels)[order]
    else:
        sorted_labels = None

    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap = ax.imshow(features, aspect="auto", cmap="coolwarm")
    ax.set_title(title)
    ax.set_xlabel("Features")
    ax.set_ylabel("Samples")

    num_samples, num_features = features.shape
    y_ticks = _select_tick_positions(num_samples, max_ticks)
    x_ticks = _select_tick_positions(num_features, max_ticks)

    if feature_names and len(feature_names) == num_features:
        x_labels = [feature_names[idx] for idx in x_ticks]
    else:
        x_labels = [f"f{idx}" for idx in x_ticks]

    if show_sample_labels and sample_names and len(sample_names) == num_samples:
        y_labels = [sample_names[idx] for idx in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=6)
    else:
        ax.set_yticks([])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=6, rotation=90)

    if sorted_labels is not None and num_samples > 1:
        changes = np.where(np.diff(sorted_labels) != 0)[0]
        for idx in changes:
            ax.axhline(idx + 0.5, color="black", linewidth=0.5, alpha=0.6)

    fig.colorbar(heatmap, ax=ax, fraction=0.035, pad=0.02)
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
    parser = argparse.ArgumentParser(description="Run Agglomerative Clustering over image features.")
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default=DEFAULT_FEATURE_SET,
        help="Feature set to use: set1 (color stats), set2 (HOG), set3 (BoVW), set4 (LBP).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_set = args.feature_set

    samples = discover_samples(DATA_ROOT, SPLIT)
    if not samples:
        raise RuntimeError(f"No samples found under {DATA_ROOT / SPLIT}")
    sample_names = [f"{sample.label}/{sample.path.stem}" for sample in samples]

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
    features_scaled_cache: dict[int, tuple[np.ndarray, list[str]]] = {}
    features_cache: dict[tuple[int, int | None], tuple[np.ndarray, np.ndarray, list[str]]] = {}

    def get_scaled_features(bins: int) -> tuple[np.ndarray, list[str]]:
        cached = features_scaled_cache.get(bins)
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

        if features.size == 0:
            raise RuntimeError("Feature extraction returned an empty matrix.")

        features_scaled = StandardScaler().fit_transform(features)
        features_scaled_cache[bins] = (features_scaled, labels)
        return features_scaled, labels

    def get_features_for_bins(
        bins: int, pca_components: int | None
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        cache_key = (bins, pca_components)
        cached = features_cache.get(cache_key)
        if cached is not None:
            return cached

        features_scaled, labels = get_scaled_features(bins)
        features_cluster = reduce_for_clustering(features_scaled, pca_components)
        if POST_PCA_STANDARDIZE and features_cluster.size:
            features_cluster = StandardScaler().fit_transform(features_cluster)

        if features_cluster.shape[1] >= 2:
            x_2d = PCA(n_components=2, random_state=42).fit_transform(features_cluster)
        else:
            x_2d = np.column_stack([features_cluster[:, 0], np.zeros(len(features_cluster))])

        features_cache[cache_key] = (features_cluster, x_2d, labels)
        return features_cluster, x_2d, labels

    if feature_set == "set1":
        results_path = output_dir / f"ahc_{SPLIT}_summary.csv"
    else:
        results_path = output_dir / f"ahc_{SPLIT}_{feature_set}_summary.csv"
    summary_columns = build_summary_columns(feature_set)
    metric_columns = [
        "pca_components",
        "n_clusters",
        "linkage",
        "metric",
        "clusters",
        "accuracy",
        "accuracy_binary",
        "precision",
        "recall",
        "f2",
        "ari",
        "nmi",
    ]
    rows = [",".join(summary_columns + metric_columns) + "\n"]

    if PLOT_HEATMAP:
        for bins in bins_values:
            features_scaled, _ = get_scaled_features(bins)
            heatmap_path = output_dir / (
                f"ahc_{SPLIT}_{feature_tag(bins, feature_set)}_features_heatmap.png"
            )
            title = f"Feature heatmap ({feature_set}, {feature_tag(bins, feature_set)})"
            plot_feature_heatmap(
                features_scaled,
                sample_names,
                heatmap_path,
                title=title,
                show_sample_labels=False,
            )
            print(f"Saved feature heatmap to {heatmap_path}")

    for config in configs:
        bins = int(config["bins"])
        pca_components = config.get("pca_components", CLUSTER_PCA_COMPONENTS)
        n_clusters = int(config["n_clusters"])
        linkage = str(config["linkage"])
        metric = str(config["metric"])

        features_cluster, x_2d, labels = get_features_for_bins(bins, pca_components)
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
            f"ahc_{SPLIT}_{feature_tag(bins, feature_set)}_{pca_tag(pca_components)}_n{n_clusters}_link{linkage}_{metric}_pca.png"
        )
        plot_clusters(x_2d, cluster_labels, output_path)

        if PLOT_CLUSTERED_HEATMAP:
            features_scaled, _ = get_scaled_features(bins)
            clustered_heatmap_path = output_dir / (
                f"ahc_{SPLIT}_{feature_tag(bins, feature_set)}_{pca_tag(pca_components)}_n{n_clusters}_link{linkage}_{metric}_clusters_heatmap.png"
            )
            title = (
                f"Clustered feature heatmap ({feature_set}, n={n_clusters}, {linkage})"
            )
            plot_feature_heatmap(
                features_scaled,
                sample_names,
                clustered_heatmap_path,
                title=title,
                show_sample_labels=False,
                cluster_labels=cluster_labels,
            )
            print(f"Saved clustered feature heatmap to {clustered_heatmap_path}")

        cluster_count = len(np.unique(cluster_labels))
        accuracy = hungarian_match_accuracy(labels, cluster_labels)
        accuracy_binary = binary_accuracy(labels, cluster_labels, HEALTHY_LABELS)
        mapped_labels = hungarian_match_predict(labels, cluster_labels)
        class_labels = sorted(set(labels))
        precision, recall, _, _ = precision_recall_fscore_support(
            labels,
            mapped_labels,
            labels=class_labels,
            average="macro",
            zero_division=0,
        )
        f2 = fbeta_score(
            labels,
            mapped_labels,
            labels=class_labels,
            beta=2,
            average="macro",
            zero_division=0,
        )
        ari = adjusted_rand_score(labels, cluster_labels)
        nmi = normalized_mutual_info_score(labels, cluster_labels)
        rows.append(
            ",".join(
                build_summary_prefix(feature_set, bins)
                + [
                    str(pca_components),
                    str(n_clusters),
                    linkage,
                    metric,
                    str(cluster_count),
                    f"{accuracy:.4f}",
                    f"{accuracy_binary:.4f}",
                    f"{precision:.4f}",
                    f"{recall:.4f}",
                    f"{f2:.4f}",
                    f"{ari:.4f}",
                    f"{nmi:.4f}",
                ]
            )
            + "\n"
        )
        print(f"Saved plot to {output_path}")

        report_path = output_dir / (
            f"ahc_{SPLIT}_{feature_tag(bins, feature_set)}_{pca_tag(pca_components)}_n{n_clusters}_link{linkage}_{metric}_clusters.csv"
        )
        report_rows = build_cluster_report(labels, cluster_labels)
        report_path.write_text("".join(report_rows), encoding="utf-8")
        print(f"Saved cluster report to {report_path}")

    results_path.write_text("".join(rows), encoding="utf-8")
    print(f"Saved summary to {results_path}")


if __name__ == "__main__":
    main()
