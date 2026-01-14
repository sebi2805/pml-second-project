from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: str
    split: str


def iter_samples(data_root: Path, split: str) -> Iterable[ImageSample]:
    split_dir = data_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label = class_dir.name
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield ImageSample(path=image_path, label=label, split=split)


def discover_samples(data_root: Path, split: str) -> List[ImageSample]:
    return list(iter_samples(data_root, split))


def load_image(path: Path, resize: Optional[Tuple[int, int]] = None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    if resize is not None:
        image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
    return image
