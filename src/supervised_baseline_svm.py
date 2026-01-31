import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from data_io import gather_images
from features import build_feature_matrix

DATA_ROOT = Path("data")
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
BINS = 32
RESIZE = (128, 128)
DEFAULT_FEATURE_SET = "set2"
FEATURE_SETS = ("set1", "set2", "set4")
HOG_PARAMS = {"orientations": 9, "pixels_per_cell": (8, 8), "cells_per_block": (2, 2)}
LBP_PARAMS = {"radius": 1, "n_points": 8, "method": "uniform"}
BINARY_POSITIVE_LABELS = {"Healthy_Nail"}

MAX_ITER = 100
OUTPUT_CSV = Path("output/supervised_baseline_svm_results.csv")


# in the spirit of all the other parts i will iterate manually over the grid
GRID_CONFIG = {
    "svm": "svc",
    "params": [
        {"kernel": ["linear"], "C": [0.1, 1, 10]},
        {"kernel": ["rbf"], "C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
        # the poly one is most expensive in terms of time, you might want to comment it out
        {
            "kernel": ["poly"],
            "C": [0.1, 1, 10],
            "gamma": ["scale"],
            "degree": [2, 3],
            "coef0": [0.0, 1.0],
        },
    ],
}


def build_features(samples, feature_set, bins):
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

    return features, labels


def to_binary(labels):
    binary = []
    for label in labels:
        if label in BINARY_POSITIVE_LABELS:
            binary.append(1)
        else:
            binary.append(0)
    return binary


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

    train_samples = gather_images(DATA_ROOT, TRAIN_SPLIT)
    val_samples = gather_images(DATA_ROOT, VAL_SPLIT)

    x_train, y_train = build_features(train_samples, args.feature_set, BINS)
    x_val, y_val = build_features(val_samples, args.feature_set, BINS)

    # i ddint apply pca here because svm can handle high dimensional data pretty well
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    results = []
    best = None
    grid_config = GRID_CONFIG
    param_grid = grid_config["params"]

    for params in ParameterGrid(param_grid):
        model = SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            degree=params.get("degree", 3),
            coef0=params.get("coef0", 0.0),
            max_iter=MAX_ITER,
        )
        row = {
            "svm": "svc",
            "kernel": params.get("kernel", "rbf"),
            "c": params.get("C", 1.0),
            "gamma": params.get("gamma"),
            "degree": params.get("degree"),
            "coef0": params.get("coef0"),
            "loss": None,
        }

        model.fit(x_train, y_train)
        preds = model.predict(x_val)

        acc = accuracy_score(y_val, preds)
        f1_macro = f1_score(y_val, preds, average="macro")

        # here i want to map the classes in binary
        binary_acc = accuracy_score(to_binary(y_val), to_binary(list(preds)))

        row["acc"] = acc
        row["f1_macro"] = f1_macro
        row["binary_acc"] = binary_acc
        results.append(row)
        print(row)
        
        # i want to improve g1, i could improve accuracy but the datasets is not really balanced
        if best is None or f1_macro > best["f1_macro"]:
            best = row

    print("best svm configuration")
    print(best)

    output_path = OUTPUT_CSV
    if output_path.parent:
        output_path.parent.mkdir(exist_ok=True)
    rows = [
        "feature_set,svm,kernel,c,gamma,degree,coef0,loss,max_iter,acc,f1_macro,"
        "binary_acc,binary_positive_labels,train_samples,val_samples,is_best\n"
    ]
    for row in results:
        is_best = row is best
        gamma_value = row["gamma"]
        degree_value = row["degree"]
        coef0_value = row["coef0"]
        loss_value = row["loss"]
        rows.append(
            f"{args.feature_set},{row['svm']},{row['kernel']},{row['c']},"
            f"{gamma_value},"
            f"{degree_value},"
            f"{coef0_value},"
            f"{loss_value},"
            f"{MAX_ITER},{row['acc']:.6f},{row['f1_macro']:.6f},"
            f"{row['binary_acc']:.6f},{'|'.join(sorted(BINARY_POSITIVE_LABELS))},"
            f"{len(y_train)},{len(y_val)},{int(is_best)}\n"
        )
    output_path.write_text("".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
