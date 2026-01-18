from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _hungarian_minimize(cost: np.ndarray) -> list[int]:
    cost = np.asarray(cost, dtype=float)
    n_rows, n_cols = cost.shape
    if n_rows == 0:
        return []
    if n_rows > n_cols:
        raise ValueError("Cost matrix must have rows <= columns.")

    u = [0.0] * (n_rows + 1)
    v = [0.0] * (n_cols + 1)
    p = [0] * (n_cols + 1)
    way = [0] * (n_cols + 1)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n_cols + 1)
        used = [False] * (n_cols + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n_rows
    for j in range(1, n_cols + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def _hungarian_max_sum(matrix: np.ndarray) -> int:
    if matrix.size == 0:
        return 0
    n_rows, n_cols = matrix.shape
    size = max(n_rows, n_cols)
    padded = np.zeros((size, size), dtype=float)
    padded[:n_rows, :n_cols] = matrix

    max_value = float(np.max(padded)) if padded.size else 0.0
    cost = max_value - padded
    assignment = _hungarian_minimize(cost)

    total = 0.0
    for i in range(n_rows):
        j = assignment[i]
        if 0 <= j < n_cols:
            total += padded[i, j]
    return int(total)


def _hungarian_max_assignment(matrix: np.ndarray) -> list[int]:
    if matrix.size == 0:
        return []
    n_rows, n_cols = matrix.shape
    size = max(n_rows, n_cols)
    padded = np.zeros((size, size), dtype=float)
    padded[:n_rows, :n_cols] = matrix

    max_value = float(np.max(padded)) if padded.size else 0.0
    cost = max_value - padded
    assignment = _hungarian_minimize(cost)

    mapped = []
    for i in range(n_rows):
        j = assignment[i]
        if j is None or j < 0 or j >= n_cols:
            mapped.append(-1)
        else:
            mapped.append(j)
    return mapped


def hungarian_match_mapping(
    labels_true: Sequence[str] | Iterable[str],
    labels_pred: Sequence[int] | Iterable[int],
) -> dict[int, str]:
    y_true = list(labels_true)
    y_pred = list(labels_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("labels_true and labels_pred must be the same length.")
    if not y_true:
        return {}

    true_labels = sorted(set(y_true))
    pred_labels = sorted(set(y_pred))
    true_index = {label: idx for idx, label in enumerate(true_labels)}
    pred_index = {label: idx for idx, label in enumerate(pred_labels)}

    matrix = np.zeros((len(pred_labels), len(true_labels)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred, strict=False):
        matrix[pred_index[pred_label], true_index[true_label]] += 1

    assignment = _hungarian_max_assignment(matrix)
    mapping: dict[int, str] = {}
    for pred_label, col_idx in zip(pred_labels, assignment, strict=False):
        if 0 <= col_idx < len(true_labels):
            mapping[int(pred_label)] = true_labels[col_idx]
    return mapping


def apply_label_mapping(
    labels_pred: Sequence[int] | Iterable[int],
    mapping: dict[int, str],
    fallback_label: str = "__unassigned__",
) -> list[str]:
    return [mapping.get(int(label), fallback_label) for label in labels_pred]


def hungarian_match_counts(
    labels_true: Sequence[str] | Iterable[str],
    labels_pred: Sequence[int] | Iterable[int],
) -> tuple[int, int]:
    y_true = list(labels_true)
    y_pred = list(labels_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("labels_true and labels_pred must be the same length.")
    if not y_true:
        return 0, 0

    true_labels = sorted(set(y_true))
    pred_labels = sorted(set(y_pred))
    true_index = {label: idx for idx, label in enumerate(true_labels)}
    pred_index = {label: idx for idx, label in enumerate(pred_labels)}

    matrix = np.zeros((len(pred_labels), len(true_labels)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred, strict=False):
        matrix[pred_index[pred_label], true_index[true_label]] += 1

    correct = _hungarian_max_sum(matrix)
    return int(correct), len(y_true)


def hungarian_match_accuracy(
    labels_true: Sequence[str] | Iterable[str],
    labels_pred: Sequence[int] | Iterable[int],
) -> float:
    correct, total = hungarian_match_counts(labels_true, labels_pred)
    if total == 0:
        return 0.0
    return correct / total


def hungarian_match_predict(
    labels_true: Sequence[str] | Iterable[str],
    labels_pred: Sequence[int] | Iterable[int],
    fallback_label: str = "__unassigned__",
) -> list[str]:
    mapping = hungarian_match_mapping(labels_true, labels_pred)
    return apply_label_mapping(labels_pred, mapping, fallback_label=fallback_label)


def binary_match_counts(
    labels_true: Sequence[str] | Iterable[str],
    labels_pred: Sequence[int] | Iterable[int],
    positive_labels: set[str],
) -> tuple[int, int]:
    y_true = list(labels_true)
    y_pred = list(labels_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("labels_true and labels_pred must be the same length.")
    if not y_true:
        return 0, 0

    cluster_counts: dict[int, list[int]] = {}
    for true_label, pred_label in zip(y_true, y_pred, strict=False):
        counts = cluster_counts.setdefault(int(pred_label), [0, 0])
        if true_label in positive_labels:
            counts[1] += 1
        else:
            counts[0] += 1

    correct = sum(max(counts) for counts in cluster_counts.values())
    return int(correct), len(y_true)


def binary_accuracy(
    labels_true: Sequence[str] | Iterable[str],
    labels_pred: Sequence[int] | Iterable[int],
    positive_labels: set[str],
) -> float:
    correct, total = binary_match_counts(labels_true, labels_pred, positive_labels)
    if total == 0:
        return 0.0
    return correct / total
