from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from cluster_metrics import binary_match_counts, majority_vote_counts
from cluster_utils import (
    build_cluster_features,
    build_cluster_report,
    build_eval_features,
    build_summary_prefix,
    plot_clusters,
)
from data_io import gather_images
from pipeline_utils import (
    build_output_dir,
    build_results_path,
    init_summary_rows,
    parse_feature_set_args,
)


# some of the comments are common between adhc and dbscan because i externalized the comon code
# as much as I could, but some of the parts reamin the samen

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
    "eps": np.linspace(0.5, 2, 10),
    # "eps": np.linspace(0.5, 7, 100),
    # "min_samples": [3, 5, 8, 12],
    "min_samples": [3],
    "metric": ["euclidean"],
    "pca_components": [CLUSTER_PCA_COMPONENTS],
}


# tje same function as in ahc, adapted for dbscan
def get_configs(feature_set):
    configs = []
    bins_values = GRID_SEARCH["bins"] if feature_set == "set1" else [None]
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



# i want to see the % of noise, because we can say they are "labeled" and can help accuracy
def summarize_labels(labels):
    unique, counts = np.unique(labels, return_counts=True)
    cluster_count = np.sum(unique != -1)
    noise_count = counts[unique == -1][0] if np.any(unique == -1) else 0
    return cluster_count, noise_count



def format_dbscan_metrics(labels_true, cluster_labels):
    total_samples = len(cluster_labels)

    # i want to remove the noise
    clustered_mask = cluster_labels != -1
    labels_array = np.asarray(labels_true)

    # then only the labels that have a proper cluster will decide the accuracy
    clustered_true = labels_array[clustered_mask]
    clustered_pred = cluster_labels[clustered_mask]
    correct_clustered, _ = majority_vote_counts(clustered_true, clustered_pred)


    accuracy_overall = correct_clustered / total_samples
    correct_binary, _ = binary_match_counts(
        clustered_true, clustered_pred, HEALTHY_LABELS
    )
    accuracy_binary_overall = (
        correct_binary / total_samples
    )
    ari_overall = adjusted_rand_score(labels_array, cluster_labels)
    nmi_overall = normalized_mutual_info_score(labels_array, cluster_labels)

    return [
        f"{accuracy_overall:.4f}",
        f"{accuracy_binary_overall:.4f}",
        f"{ari_overall:.4f}",
        f"{nmi_overall:.4f}",
    ]


def main():
    args = parse_feature_set_args(
        FEATURE_SETS,
        DEFAULT_FEATURE_SET,
        "Run DBSCAN over image features.",
    )
    feature_set = args.feature_set
    train_split = "train"
    eval_split = "validation"

    train_samples = gather_images(DATA_ROOT, train_split)
    eval_samples = None
    eval_samples = gather_images(DATA_ROOT, eval_split)

    output_dir = build_output_dir(OUTPUT_DIR, "dbscan", feature_set)

    configs = get_configs(feature_set)
    bins_values = list({config["bins"] for config in configs})
    pca_values = list({config.get("pca_components") for config in configs})

    for bins in bins_values:
        for pca_components in pca_values:
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
            features_cluster, _, _, _, _, _ = cluster_data
        

    results_path = build_results_path(output_dir, "dbscan", train_split, feature_set)
    metric_columns = [
        "pca_components",
        "eps",
        "min_samples",
        "metric",
        "accuracy_overall",
        "accuracy_binary_overall",
        "ari_overall",
        "nmi_overall",
    ]
    metric_columns.extend(
        [
            "eval_accuracy_overall",
            "eval_accuracy_binary_overall",
            "eval_ari_overall",
            "eval_nmi_overall",
        ]
    )
    rows = init_summary_rows(feature_set, metric_columns)


    # in case the pca components are the same we can reuse the features
    # but for experiments we recompute 
    for config in configs:
        bins = config["bins"]
        pca_components = config.get("pca_components", CLUSTER_PCA_COMPONENTS)
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

        dbscan = DBSCAN(
            eps=config["eps"],
            min_samples=config["min_samples"],
            metric=config["metric"],
        )
        cluster_labels = dbscan.fit_predict(features_cluster)

        eps = config["eps"]
        min_samples = config["min_samples"]
        metric = config["metric"]
        output_path = output_dir / (
            f"dbscan_{train_split}_{feature_tag_value}_{pca_tag_value}_eps{eps}_min{min_samples}_{metric}_pca.png"
        )
        
        plot_clusters(
            x_2d,
            cluster_labels,
            output_path,
            "DBSCAN clusters (PCA 2d)",
            noise_label=-1,
        )

        metrics = format_dbscan_metrics(labels, cluster_labels)
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
        eval_dbscan = DBSCAN(
            eps=config["eps"],
            min_samples=config["min_samples"],
            metric=config["metric"],
        )

        # dbscan does not have predict, we could try to assign based on nearest neighbors
        # and then filter if its<eps, but i dont think that is purpose of dbscan, because maybe i have labels that
        # are deemed as noise but with validation it will form a cluster
        eval_cluster_labels = eval_dbscan.fit_predict(eval_features)
        eval_metrics = format_dbscan_metrics(eval_labels, eval_cluster_labels)

        row_values = (
            build_summary_prefix(feature_set, bins, HOG_PARAMS, LBP_PARAMS)
            + [
                pca_components,
                eps,
                min_samples,
                metric,
            ]
            + metrics
            + eval_metrics
        )
        rows.append(",".join(map(str, row_values)) + "\n")

        report_path = output_dir / (
            f"dbscan_{train_split}_{feature_tag_value}_{pca_tag_value}_eps{eps}_min{min_samples}_{metric}_clusters.csv"
        )
        report_rows = build_cluster_report(labels, cluster_labels)
        report_path.write_text("".join(report_rows), encoding="utf-8")

    results_path.write_text("".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
