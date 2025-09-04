import json
import random
from pathlib import Path
from typing import Dict

from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np


def swap_left_right(label: str) -> str:
    """Swap '-left' and '-right' in the class label if present."""
    if "-left" in label:
        return label.replace("-left", "-right")
    if "-right" in label:
        return label.replace("-right", "-left")
    return label


def _pil_to_cv(img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV RGB ndarray."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR ndarray back to PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def augment_image(img: Image.Image, aug_name: str):
    """Apply the augmentation specified by `aug_name` to `img`."""
    if aug_name == "flip":
        return ImageOps.mirror(img)

    if aug_name == "rotate":
        # Random rotation between -10 and 10 degrees (excluding 0)  
        angle = random.choice(list(range(-10, 0)) + list(range(1, 11)))
        return img.rotate(angle, expand=True, fillcolor=(0, 0, 0))

    if aug_name == "crop":
        # Random crop keeping 90-100% of the original area
        width, height = img.size
        scale = random.uniform(0.9, 1.0)
        new_w, new_h = int(width * scale), int(height * scale)
        left = random.randint(0, width - new_w)
        top = random.randint(0, height - new_h)
        return img.crop((left, top, left + new_w, top + new_h)).resize((width, height))

    if aug_name == "brightness":
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.8, 1.2)
        return enhancer.enhance(factor)

    # ---------------- Photometric / Color jitter ----------------
    if aug_name == "color_jitter":
        # Randomly adjust brightness, contrast, saturation
        result = img
        result = ImageEnhance.Brightness(result).enhance(random.uniform(0.8, 1.2))
        result = ImageEnhance.Contrast(result).enhance(random.uniform(0.8, 1.2))
        result = ImageEnhance.Color(result).enhance(random.uniform(0.8, 1.2))

        # Random hue shift (via HSV)
        hue_shift = random.randint(-10, 10)  # OpenCV HSV hue range: 0-179
        hsv = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + hue_shift) % 180
        result = Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        return result

    # ---------------- CLAHE (Adaptive Histogram Equalization) ----------------
    if aug_name == "clahe":
        lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab = cv2.merge([l2, a, b])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)

    # ---------------- Noise ----------------
    if aug_name == "gaussian_noise":
        arr = np.array(img).astype(np.float32)
        sigma = random.uniform(5, 20)
        noise = np.random.normal(0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    if aug_name == "sp_noise":
        arr = np.array(img)
        p = random.uniform(0.005, 0.01)
        mask = np.random.choice([0, 1, 2], size=arr.shape[:2], p=[1 - 2 * p, p, p])
        arr[mask == 1] = 255  # salt
        arr[mask == 2] = 0    # pepper
        return Image.fromarray(arr)

    # ---------------- Filtering ----------------
    if aug_name == "gaussian_blur":
        arr = _pil_to_cv(img)
        k = random.choice([3, 5])
        arr = cv2.GaussianBlur(arr, (k, k), sigmaX=0)
        return _cv_to_pil(arr)

    if aug_name == "sharpen":
        arr = _pil_to_cv(img)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        arr = cv2.filter2D(arr, -1, kernel)
        return _cv_to_pil(arr)

    raise ValueError(f"Unknown augmentation: {aug_name}")


def run(
    images_dir: Path = Path("./dataset/train/images"),
    labels_path: Path = Path("./dataset/train/cls.json"),
    output_dir: Path = Path("./dataset/augmented"),
    output_labels_path: Path = Path("./dataset/augmentation/cls_augmented.json"),
):
    """Create augmented dataset based on the specified augmentations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with labels_path.open("r", encoding="utf-8") as f:
        labels: Dict[str, str] = json.load(f)

    augmented_labels: Dict[str, str] = {}

    augmentations = [
        "flip",
        "rotate",
        "crop",
        "brightness",
        "color_jitter",
        "clahe",
        "gaussian_noise",
        "sp_noise",
        "gaussian_blur",
        "sharpen",
    ]

    for img_name, label in labels.items():
        img_path = images_dir / img_name
        if not img_path.exists():
            print(f"[Warning] Missing image file: {img_path}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Error] Cannot open {img_path}: {e}")
            continue

        for aug in augmentations:
            new_img = augment_image(img, aug)
            new_name = f"{img_path.stem}_{aug}{img_path.suffix}"
            new_path = output_dir / new_name
            new_img.save(new_path, format="PNG")
            if aug == "flip":
                new_label = swap_left_right(label)
            else:
                new_label = label
            augmented_labels[new_name] = new_label

    with output_labels_path.open("w", encoding="utf-8") as f:
        json.dump(augmented_labels, f, indent=2, ensure_ascii=False)

    print(f"Augmentation complete. Generated {len(augmented_labels)} images at {output_dir}.")


if __name__ == "__main__":
    run() 