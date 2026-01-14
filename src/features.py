from __future__ import annotations

from typing import Iterable, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

from data_io import ImageSample, load_image


def _resize_if_needed(image: np.ndarray, resize: Optional[Tuple[int, int]]) -> np.ndarray:
    if resize is None:
        return image
    return cv2.resize(image, resize, interpolation=cv2.INTER_AREA)


def _channel_stats(values: np.ndarray) -> tuple[float, float, float]:
    flat = values.reshape(-1).astype(np.float32)
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    if std < 1e-6:
        skew = 0.0
    else:
        centered = flat - mean
        skew = float(np.mean(centered**3) / (std**3))
    return mean, std, skew


def extract_feature_set_1(image_bgr: np.ndarray, bins: int = 32) -> np.ndarray:
    """HSV histograms + per-channel mean/std/skewness."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    hist_features = []
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [bins], [0, 256])
        hist = hist.astype(np.float32).reshape(-1)
        hist /= hist.sum() + 1e-8
        hist_features.append(hist)

    stats = []
    for channel in range(3):
        stats.extend(_channel_stats(hsv[..., channel]))

    return np.concatenate([*hist_features, np.array(stats, dtype=np.float32)])


def extract_feature_set_2(
    image_bgr: np.ndarray,
    resize: Optional[Tuple[int, int]] = None,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (8, 8),
    cells_per_block: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    """HOG descriptor on a resized grayscale image."""
    image_bgr = _resize_if_needed(image_bgr, resize)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    cell_w, cell_h = pixels_per_cell
    block_w, block_h = cells_per_block

    win_w = (width // cell_w) * cell_w
    win_h = (height // cell_h) * cell_h
    if win_w == 0 or win_h == 0:
        return np.empty((0,), dtype=np.float32)

    block_size = (block_w * cell_w, block_h * cell_h)
    if block_size[0] > win_w or block_size[1] > win_h:
        return np.empty((0,), dtype=np.float32)

    gray = gray[:win_h, :win_w]
    block_stride = (cell_w, cell_h)
    hog = cv2.HOGDescriptor(
        (win_w, win_h), block_size, block_stride, (cell_w, cell_h), orientations
    )
    descriptors = hog.compute(gray)
    if descriptors is None:
        return np.empty((0,), dtype=np.float32)

    return descriptors.reshape(-1).astype(np.float32)


def _extract_orb_descriptors(image_bgr: np.ndarray, max_features: int) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max(1, max_features))
    _, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None:
        return np.empty((0, 32), dtype=np.uint8)
    return descriptors


def build_orb_codebook(
    samples: Iterable[ImageSample],
    num_words: int = 64,
    max_features: int = 500,
    max_descriptors: int = 10_000,
    resize: Optional[Tuple[int, int]] = None,
    random_state: int = 42,
) -> np.ndarray:
    descriptors_list = []
    for sample in samples:
        image = load_image(sample.path, resize=resize)
        descriptors = _extract_orb_descriptors(image, max_features=max_features)
        if descriptors.size:
            descriptors_list.append(descriptors)

    if not descriptors_list:
        raise ValueError("No ORB descriptors found to build a codebook.")

    all_descriptors = np.vstack(descriptors_list).astype(np.float32)
    if all_descriptors.shape[0] > max_descriptors:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(all_descriptors.shape[0], size=max_descriptors, replace=False)
        all_descriptors = all_descriptors[indices]

    if all_descriptors.shape[0] < num_words:
        raise ValueError(
            f"Not enough descriptors ({all_descriptors.shape[0]}) for {num_words} words."
        )

    kmeans = KMeans(n_clusters=num_words, random_state=random_state, n_init="auto")
    kmeans.fit(all_descriptors)
    return kmeans.cluster_centers_.astype(np.float32)


def extract_feature_set_3(
    image_bgr: np.ndarray,
    codebook: np.ndarray,
    resize: Optional[Tuple[int, int]] = None,
    max_features: int = 500,
) -> np.ndarray:
    """Bag of visual words histogram using ORB descriptors."""
    if codebook.size == 0:
        raise ValueError("Codebook is empty.")

    image_bgr = _resize_if_needed(image_bgr, resize)
    descriptors = _extract_orb_descriptors(image_bgr, max_features=max_features)
    num_words = int(codebook.shape[0])
    if descriptors.size == 0:
        return np.zeros(num_words, dtype=np.float32)

    assignments = pairwise_distances_argmin(
        descriptors.astype(np.float32), codebook.astype(np.float32)
    )
    hist = np.bincount(assignments, minlength=num_words).astype(np.float32)
    hist /= hist.sum() + 1e-8
    return hist


def build_feature_matrix(
    samples: Iterable[ImageSample],
    bins: int = 32,
    resize: Optional[Tuple[int, int]] = None,
    feature_set: str = "set1",
    hog_params: Optional[dict[str, int | Tuple[int, int]]] = None,
    codebook: Optional[np.ndarray] = None,
    orb_params: Optional[dict[str, int]] = None,
) -> tuple[np.ndarray, list[str]]:
    features = []
    labels = []
    hog_params = hog_params or {}
    orb_params = orb_params or {}
    for sample in samples:
        if feature_set == "set1":
            image = load_image(sample.path, resize=resize)
            features.append(extract_feature_set_1(image, bins=bins))
        elif feature_set == "set2":
            image = load_image(sample.path, resize=None)
            features.append(
                extract_feature_set_2(image, resize=resize, **hog_params)
            )
        elif feature_set == "set3":
            if codebook is None:
                raise ValueError("codebook is required for feature_set='set3'.")
            image = load_image(sample.path, resize=None)
            features.append(
                extract_feature_set_3(image, codebook=codebook, resize=resize, **orb_params)
            )
        else:
            raise ValueError(f"Unknown feature_set: {feature_set}")
        labels.append(sample.label)

    if not features:
        return np.empty((0, 0), dtype=np.float32), labels

    return np.vstack(features), labels
