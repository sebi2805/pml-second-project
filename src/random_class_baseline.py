from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from data_io import gather_images

DATA_ROOT = Path("data")
RUNS = 50
DEFAULT_SEED = 2805
HEALTHY_LABELS = {"Healthy_Nail"}


# we could run the random only once, but then the results would be less stable
# thats why i run it for multiple times and take the mean

def binary_accuracy(y_true, y_pred, healthy_labels):
    healthy_list = list(healthy_labels)
    true_healthy = np.isin(y_true, healthy_list)
    pred_healthy = np.isin(y_pred, healthy_list)

    return accuracy_score(true_healthy, pred_healthy)


def run_random_class_baseline(split, runs, seed):
    # i dont want to upload the images in memory bcs i only need the label
    samples = gather_images(DATA_ROOT, split)

    labels_list = []
    for sample in samples:
        labels_list.append(sample["label"])
    labels = np.array(labels_list, dtype=str)

    classes_set = set()
    for label in labels:
        classes_set.add(label)
    classes = sorted(classes_set)

    overall_accs = []
    binary_accs = []

    for run in range(runs):
        rng = np.random.default_rng(seed + run)
        y_pred = rng.choice(classes, size=labels.size, replace=True)

        overall_accs.append(np.mean(y_pred == labels))
        binary_accs.append(binary_accuracy(labels, y_pred, HEALTHY_LABELS))

    overall_mean = np.mean(overall_accs)
    binary_mean = np.mean(binary_accs)

    print(f"overall accuracy mean {overall_mean:.4f}")
    print(f"binary accuracy mean {binary_mean:.4f}")


def main():
    run_random_class_baseline("validation", RUNS, DEFAULT_SEED)


if __name__ == "__main__":
    main()
