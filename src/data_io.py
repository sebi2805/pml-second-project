import cv2


def gather_images(data_root, split):
    split_dir = data_root / split
    class_labels = [
        "Acral_Lentiginous_Melanoma",
        "blue_finger",
        "clubbing",
        "Healthy_Nail",
        "Onychogryphosis",
        "pitting",
    ]
    samples = []
    for label in class_labels:
        class_dir = split_dir / label

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in {".jpg", ".jpeg"}:
                samples.append(
                    {
                        "path": image_path,
                        "label": label,
                        "split": split,
                    }
                )
    return samples


def load_image(path, resize):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    # sometimes the resize is null when i tried eda
    if resize is not None:
        image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)

    return image
