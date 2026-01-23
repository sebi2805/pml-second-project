from pathlib import Path
import argparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from cluster_metrics import (
    binary_accuracy,
    majority_vote_accuracy,
)
from cluster_utils import (
    build_cluster_features,
    build_cluster_report,
    build_eval_features,
    build_summary_columns,
    build_summary_prefix,
    plot_clusters,
)
from data_io import gather_images

DATA_ROOT = Path("data")
OUTPUT_DIR = Path("output")
RESIZE = (128, 128)
DEFAULT_FEATURE_SET = "set2"
FEATURE_SETS = ("set1", "set2", "set4")
HOG_PARAMS = {"orientations": 9, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2)}
LBP_PARAMS = {"radius": 1, "n_points": 8, "method": "uniform"}

CLUSTER_PCA_COMPONENTS = 20
HEALTHY_LABELS = {"Healthy_Nail"}

GRID_SEARCH = {
    "bins": [16, 32, 64],
    "n_clusters": [4, 6, 8, 10],
    "linkage": ["ward", "average", "complete"],
    "metric": ["euclidean", "manhattan"],
    "pca_components": [CLUSTER_PCA_COMPONENTS],
}


def get_configs(feature_set):
    configs = []
    bins_values = GRID_SEARCH["bins"] if feature_set == "set1" else []
    for bins in bins_values:
        for n_clusters in GRID_SEARCH["n_clusters"]:
            for linkage in GRID_SEARCH["linkage"]:
                for metric in GRID_SEARCH["metric"]:

                    # this is one specific contraint for ahc
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default=DEFAULT_FEATURE_SET,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    feature_set = args.feature_set
    train_split = "train"
    eval_split = "validation"

    train_samples = gather_images(DATA_ROOT, train_split)
    eval_samples = gather_images(DATA_ROOT, eval_split)

    sample_names = []
    for sample in train_samples:
        sample_names.append(f"{sample['label']}/{sample['path'].stem}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_dir = OUTPUT_DIR / "ahc" / feature_set
    output_dir.mkdir(exist_ok=True)

    configs = get_configs(feature_set)

    if feature_set == "set1":
        results_path = output_dir / f"ahc_{train_split}_summary.csv"
    else:
        results_path = output_dir / f"ahc_{train_split}_{feature_set}_summary.csv"
    summary_columns = build_summary_columns(feature_set)
    metric_columns = [
        "pca_components",
        "n_clusters",
        "linkage",
        "metric",
        "accuracy",
        "accuracy_binary",
        "ari",
        "nmi",
    ]

    # jsut for eval
    metric_columns.extend(
        [
            "eval_accuracy",
            "eval_accuracy_binary",
            "eval_ari",
            "eval_nmi",
        ]
    )
    rows = [",".join(summary_columns + metric_columns) + "\n"]

    for config in configs:
        bins = config["bins"]
        pca_components = config.get("pca_components", CLUSTER_PCA_COMPONENTS)
        n_clusters = config["n_clusters"]
        linkage = config["linkage"]
        metric = config["metric"]
        feature_tag_value = f"bins{bins}" if feature_set == "set1" else feature_set
        pca_tag_value = f"pca{pca_components}"

        cluster_data = build_cluster_features(
            train_samples,
            feature_set,
            bins,
            RESIZE,
            HOG_PARAMS,
            LBP_PARAMS,
            pca_components,
        )
        (
            features_cluster,
            x_2d,
            labels,
            scaler,
            pca,
            post_scaler,
        ) = cluster_data

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
        )
        cluster_labels = model.fit_predict(features_cluster)

        output_path = output_dir / (
            f"ahc_{train_split}_{feature_tag_value}_{pca_tag_value}_n{n_clusters}_link{linkage}_{metric}_pca.png"
        )
        plot_clusters(x_2d, cluster_labels, output_path, "AHC clusters (PCA 2D)")

        accuracy = majority_vote_accuracy(labels, cluster_labels)
        accuracy_binary = binary_accuracy(labels, cluster_labels, HEALTHY_LABELS)
        ari = adjusted_rand_score(labels, cluster_labels)
        nmi = normalized_mutual_info_score(labels, cluster_labels)

        eval_data = build_eval_features(
            eval_samples,
            feature_set,
            bins,
            RESIZE,
            HOG_PARAMS,
            LBP_PARAMS,
            scaler,
            pca,
            post_scaler,
        )
        eval_features, eval_labels = eval_data
        eval_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric,
        )
        eval_cluster_labels = eval_model.fit_predict(eval_features)
        eval_accuracy = majority_vote_accuracy(eval_labels, eval_cluster_labels)
        eval_accuracy_binary = binary_accuracy(
            eval_labels, eval_cluster_labels, HEALTHY_LABELS
        )
        eval_ari = adjusted_rand_score(eval_labels, eval_cluster_labels)
        eval_nmi = normalized_mutual_info_score(eval_labels, eval_cluster_labels)
        eval_metrics = [
            f"{eval_accuracy:.4f}",
            f"{eval_accuracy_binary:.4f}",
            f"{eval_ari:.4f}",
            f"{eval_nmi:.4f}",
        ]
        row_values = (
            build_summary_prefix(feature_set, bins, HOG_PARAMS, LBP_PARAMS)
            + [
                pca_components,
                n_clusters,
                linkage,
                metric,
                f"{accuracy:.4f}",
                f"{accuracy_binary:.4f}",
                f"{ari:.4f}",
                f"{nmi:.4f}",
            ]
            + eval_metrics
        )
        rows.append(",".join(map(str, row_values)) + "\n")
        print(f"Saved plot to {output_path}")

        report_path = output_dir / (
            f"ahc_{train_split}_{feature_tag_value}_{pca_tag_value}_n{n_clusters}_link{linkage}_{metric}_clusters.csv"
        )
        report_rows = build_cluster_report(labels, cluster_labels)
        report_path.write_text("".join(report_rows), encoding="utf-8")
        print(f"Saved cluster report to {report_path}")

    results_path.write_text("".join(rows), encoding="utf-8")
    print(f"Saved summary to {results_path}")


if __name__ == "__main__":
    main()
