import argparse

from cluster_utils import build_summary_columns


# we mainly limited only to feature set , the others are controlled from constants
def parse_feature_set_args(feature_sets, default, description=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--feature-set",
        choices=feature_sets,
        default=default,
    )
    return parser.parse_args()


def build_output_dir(output_root, algorithm, feature_set):
    output_root.mkdir(exist_ok=True)
    output_dir = output_root / algorithm / feature_set
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_results_path(output_dir, algorithm, train_split, feature_set):
    if feature_set == "set1":
        return output_dir / f"{algorithm}_{train_split}_summary.csv"
    return output_dir / f"{algorithm}_{train_split}_{feature_set}_summary.csv"


def init_summary_rows(feature_set, metric_columns):
    summary_columns = build_summary_columns(feature_set)
    return [",".join(summary_columns + metric_columns) + "\n"]

