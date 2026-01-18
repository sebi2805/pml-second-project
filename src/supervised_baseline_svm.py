from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from data_io import discover_samples
from features import build_feature_matrix, build_orb_codebook

DATA_ROOT = Path("data")
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
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


def build_features(samples, feature_set: str, bins: int, codebook):
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
        raise ValueError(f"Unknown feature_set: {feature_set}")

    if features.size == 0:
        raise RuntimeError("Feature extraction returned an empty matrix.")

    return features, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a supervised SVM baseline on extracted features."
    )
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default=DEFAULT_FEATURE_SET,
        help="Feature set to use: set1 (color stats), set2 (HOG), set3 (BoVW), set4 (LBP).",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="SVM regularization strength (C).",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run a simple grid search over C values on the validation split.",
    )
    parser.add_argument(
        "--c-grid",
        type=str,
        default="0.01,0.1,1,10,100",
        help="Comma-separated C values for grid search.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Maximum iterations for LinearSVC.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to write metrics as a single-row CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_samples = discover_samples(DATA_ROOT, TRAIN_SPLIT)
    val_samples = discover_samples(DATA_ROOT, VAL_SPLIT)
    if not train_samples:
        raise RuntimeError(f"No samples found under {DATA_ROOT / TRAIN_SPLIT}")
    if not val_samples:
        raise RuntimeError(f"No samples found under {DATA_ROOT / VAL_SPLIT}")

    codebook = None
    if args.feature_set == "set3":
        codebook = build_orb_codebook(
            train_samples,
            num_words=BOVW_NUM_WORDS,
            max_features=BOVW_MAX_FEATURES,
            max_descriptors=BOVW_MAX_DESCRIPTORS,
            resize=RESIZE,
        )

    x_train, y_train = build_features(train_samples, args.feature_set, BINS, codebook)
    x_val, y_val = build_features(val_samples, args.feature_set, BINS, codebook)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    if args.grid_search:
        c_values = [value.strip() for value in args.c_grid.split(",") if value.strip()]
        if not c_values:
            raise ValueError("c_grid must include at least one value.")
        c_values_float = []
        for value in c_values:
            try:
                c_values_float.append(float(value))
            except ValueError as exc:
                raise ValueError(f"Invalid C value: {value}") from exc

        results = []
        best = None
        for c_value in c_values_float:
            clf = LinearSVC(C=c_value, max_iter=args.max_iter)
            clf.fit(x_train, y_train)
            preds = clf.predict(x_val)
            acc = accuracy_score(y_val, preds)
            f1_macro = f1_score(y_val, preds, average="macro")
            results.append((c_value, acc, f1_macro))
            if best is None or f1_macro > best[2]:
                best = (c_value, acc, f1_macro)

        if best is None:
            raise RuntimeError("Grid search failed to evaluate any C values.")

        best_c, best_acc, best_f1 = best
        print(
            "SVM grid search (LinearSVC) "
            f"feature_set={args.feature_set} "
            f"best_c={best_c} best_acc={best_acc:.4f} best_f1_macro={best_f1:.4f} "
            f"train={len(y_train)} val={len(y_val)}"
        )
        print("c,acc,f1_macro")
        for c_value, acc, f1_macro in results:
            print(f"{c_value},{acc:.4f},{f1_macro:.4f}")

        if args.output_csv is not None:
            output_path = args.output_csv
            if output_path.parent:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            rows = ["feature_set,c,max_iter,acc,f1_macro,train_samples,val_samples,is_best\n"]
            for c_value, acc, f1_macro in results:
                is_best = c_value == best_c
                rows.append(
                    f"{args.feature_set},{c_value},{args.max_iter},{acc:.6f},{f1_macro:.6f},"
                    f"{len(y_train)},{len(y_val)},{int(is_best)}\n"
                )
            output_path.write_text("".join(rows), encoding="utf-8")
            print(f"Saved supervised baseline grid results to {output_path}")
    else:
        clf = LinearSVC(C=args.c, max_iter=args.max_iter)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_val)
        acc = accuracy_score(y_val, preds)
        f1_macro = f1_score(y_val, preds, average="macro")

        print(
            "SVM baseline (LinearSVC) "
            f"feature_set={args.feature_set} "
            f"acc={acc:.4f} f1_macro={f1_macro:.4f} "
            f"train={len(y_train)} val={len(y_val)}"
        )

        if args.output_csv is not None:
            output_path = args.output_csv
            if output_path.parent:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                "feature_set,acc,f1_macro,train_samples,val_samples\n"
                f"{args.feature_set},{acc:.6f},{f1_macro:.6f},{len(y_train)},{len(y_val)}\n",
                encoding="utf-8",
            )
            print(f"Saved supervised baseline metrics to {output_path}")


if __name__ == "__main__":
    main()
