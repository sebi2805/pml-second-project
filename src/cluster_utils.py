from collections import Counter
import math

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from features import build_feature_matrix

# the code is split in building the header, the values 
# building the freq of classes in a cluster and then some util functions as the entropy


def build_summary_columns(feature_set):
    # im building the csv header and in the end the header length should match the row length
    columns = []
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
    elif feature_set == "set4":
        columns.extend(["lbp_radius", "lbp_n_points", "lbp_method"])
    return columns


def build_summary_prefix(feature_set, bins, hog_params, lbp_params):
    values = []
    if feature_set == "set1":
        values.append(bins)
    elif feature_set == "set2":
        values.extend(
            [
                hog_params["orientations"],
                hog_params["pixels_per_cell"][0],
                hog_params["pixels_per_cell"][1],
                hog_params["cells_per_block"][0],
                hog_params["cells_per_block"][1],
            ]
        )
    elif feature_set == "set4":
        values.extend(
            [
                lbp_params["radius"],
                lbp_params["n_points"],
                lbp_params["method"],
            ]
        )
    return values


def cluster_entropy(class_counts):
    # based on a cluster i want to compute how randomize are the labels
    total = sum(class_counts.values())
    entropy = 0.0
    for count in class_counts.values():
        if total == 0:
            return 0.0
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def build_cluster_report(labels_true, cluster_labels):
    #  based on the cluster i want to build what is the class distributiion
    # so i can analyze esier
    clusters = {}
    for true_label, cluster_label in zip(labels_true, cluster_labels, strict=False):
        counts = clusters.get(cluster_label)

        # if we didnt see this cluster yet
        if counts is None:
            counts = Counter()
            clusters[cluster_label] = counts
        counts[true_label] += 1

    rows = [",".join(["cluster", *labels_true]) + "\n"]
    for cluster_label in sorted(clusters):
        counts = clusters[cluster_label]
        row = ",".join(str(counts.get(label, 0)) for label in labels_true)
        rows.append(f"{cluster_label},{row}\n")
    return rows


def plot_clusters(x_2d, labels, output_path, title, noise_label=None):
    unique_labels = sorted(set(labels))
    num_labels = len(unique_labels)

    # i ddnt want to define c colormap by myself so i use tab20, that mights be a problen with dbscan
    cmap = plt.cm.get_cmap("tab20", max(1, num_labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, label in enumerate(unique_labels):
        mask = labels == label

        # i separate this for dbscan where we have a special class for noise
        if noise_label is not None and label == noise_label:
            color = "black"
            name = "noise"
        else:
            color = cmap(idx)
            name = f"cluster {label}"
        ax.scatter(
            x_2d[mask, 0],
            x_2d[mask, 1],
            s=18,
            alpha=0.75,
            color=color,
            label=name,
        )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # for dbscan it was getting messy with too many clusters
    if num_labels <= 15:
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def reduce_for_clustering(features_scaled, components, random_state=42):
    max_components = min(features_scaled.shape[0], features_scaled.shape[1])

    # in case i build too less features with HOG i want to make sure that the oca does not try to scale up
    if components is None:
        components = max_components
    else:
        components = min(components, max_components)
    if components < 1:
        components = 1
    pca = PCA(
        n_components=components,
        random_state=random_state,
    )

    return pca.fit_transform(features_scaled), pca


def build_features(samples, feature_set, bins, resize, hog_params, lbp_params):
    if feature_set == "set1":
        return build_feature_matrix(
            samples, bins=bins, resize=resize, feature_set=feature_set
        )
    if feature_set == "set2":
        return build_feature_matrix(
            samples,
            resize=resize,
            feature_set=feature_set,
            hog_params=hog_params,
        )
    return build_feature_matrix(
        samples,
        resize=resize,
        feature_set=feature_set,
        lbp_params=lbp_params,
    )


def build_cluster_features(
    samples,
    feature_set,
    bins,
    resize,
    hog_params,
    lbp_params,
    pca_components,
    # lbp handles it well without postscaling
    post_scale_feature_sets=("set1", "set2"),
):
    features, labels = build_features(
        samples, feature_set, bins, resize, hog_params, lbp_params
    )
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_cluster, pca = reduce_for_clustering(features_scaled, pca_components)

    # because the range of set1 and set2 can be quite different and big scaled
    # i want to also post scale them after pca and we need to also scale them before pca
    # because pca is sensitive to feature scaling
    post_scaler = None
    if feature_set in post_scale_feature_sets:
        post_scaler = StandardScaler()
        features_cluster = post_scaler.fit_transform(features_cluster)

    x_2d = PCA(n_components=2, random_state=2805).fit_transform(features_cluster)

    return features_cluster, x_2d, labels, scaler, pca, post_scaler


def build_eval_features(
    samples,
    feature_set,
    bins,
    resize,
    hog_params,
    lbp_params,
    scaler,
    pca,
    post_scaler,
):
    features, labels = build_features(
        samples, feature_set, bins, resize, hog_params, lbp_params
    )
    features_scaled = scaler.transform(features)
    features_cluster = pca.transform(features_scaled)

    if post_scaler is not None:
        features_cluster = post_scaler.transform(features_cluster)
    return features_cluster, labels
