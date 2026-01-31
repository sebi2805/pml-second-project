import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from data_io import load_image


def _resize(image, resize):
    return cv2.resize(image, resize, interpolation=cv2.INTER_AREA)


def extract_feature_set_1(image_bgr, bins):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    hist_features = []
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [bins], [0, 256])
        hist = hist.astype(np.float32).reshape(-1)
        hist /= hist.sum() + 1e-8
        hist_features.append(hist)

    # just to pur the 3 lists together
    return np.concatenate(hist_features)


def extract_feature_set_2(
    image_bgr,
    resize=None,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
):
    image_bgr = _resize(image_bgr, resize)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    cell_w, cell_h = pixels_per_cell
    block_w, block_h = cells_per_block

    win_w = (width // cell_w) * cell_w
    win_h = (height // cell_h) * cell_h

    block_size = (block_w * cell_w, block_h * cell_h)


    gray = gray[:win_h, :win_w]
    block_stride = (cell_w, cell_h)
    hog = cv2.HOGDescriptor(
        (win_w, win_h), block_size, block_stride, (cell_w, cell_h), orientations
    )
    descriptors = hog.compute(gray)

    return descriptors.reshape(-1).astype(np.float32)


def extract_feature_set_4(
    image_bgr,
    resize=None,
    radius=1,
    n_points=8,
    method="uniform",
):
    image_bgr = _resize(image_bgr, resize)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    points = n_points
    lbp = local_binary_pattern(gray, points, radius, method=method)

    hist, _ = np.histogram(lbp.flatten(), bins=points + 2, range=(0, points + 2))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-8
    return hist


def build_feature_matrix(
    samples,
    bins=32,
    resize=None,
    feature_set="set1",
    hog_params=None,
    lbp_params=None,
):
    features = []
    labels = []

    # we build the X and y based on samples
    for sample in samples:
        if feature_set == "set1":
            image = load_image(sample["path"], resize=resize)
            feature = extract_feature_set_1(image, bins=bins)
            features.append(feature)

        elif feature_set == "set2":
            image = load_image(sample["path"], resize=None)
            feature = extract_feature_set_2(image, resize=resize, **hog_params)
            features.append(feature)
        elif feature_set == "set4":
            image = load_image(sample["path"], resize=None)
            feature = extract_feature_set_4(image, resize=resize, **lbp_params)
            features.append(feature)

        labels.append(sample["label"])

    return np.vstack(features), labels
