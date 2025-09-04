import json
import shutil
from pathlib import Path


def merge_datasets(
    train_images_dir: Path = Path("./dataset/train/images"),
    train_labels_path: Path = Path("./dataset/train/cls_train.json"),
    aug_images_dir: Path = Path("./dataset/augmented"),
    aug_labels_path: Path = Path("./dataset/augmented/cls_augmented.json"),
    output_labels_path: Path = Path("./dataset/augmented_merge_original/cls_train.json"),
    copy_existing: bool = True,
):
    """Merge original training images/labels with augmented ones.

    Steps:
    1. Ensure `aug_images_dir` exists.
    2. Optionally copy every image listed in `train_labels_path` from `train_images_dir`
       into `aug_images_dir` (skipped if already present).
    3. Load both label dictionaries and merge them (aug labels overwrite if duplicates).
    4. Write the combined mapping to `output_labels_path` in pretty JSON format.
    """

    aug_images_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Copy original train images into augmentation directory
    # ------------------------------------------------------------------
    if copy_existing:
        missing_count = 0
        copied_count = 0
        with train_labels_path.open("r", encoding="utf-8") as f:
            train_labels = json.load(f)
        for img_name in train_labels.keys():
            src = train_images_dir / img_name
            dst = aug_images_dir / img_name
            if not src.is_file():
                missing_count += 1
                print(f"[Warning] Source image not found: {src}")
                continue
            if dst.exists():
                continue  # already copied
            shutil.copy2(src, dst)
            copied_count += 1
        print(f"Copied {copied_count} images from train set into {aug_images_dir} (missing: {missing_count})")
    else:
        with train_labels_path.open("r", encoding="utf-8") as f:
            train_labels = json.load(f)

    # ------------------------------------------------------------------
    # Step 2: Load augmented labels
    # ------------------------------------------------------------------
    with aug_labels_path.open("r", encoding="utf-8") as f:
        aug_labels = json.load(f)

    # ------------------------------------------------------------------
    # Step 3: Merge label dictionaries
    # ------------------------------------------------------------------
    combined_labels = {**train_labels, **aug_labels}
    print(f"Total records: original={len(train_labels)}, augmented={len(aug_labels)}, combined={len(combined_labels)}")

    # ------------------------------------------------------------------
    # Step 4: Save combined mapping
    # ------------------------------------------------------------------
    with output_labels_path.open("w", encoding="utf-8") as f:
        json.dump(combined_labels, f, indent=2, ensure_ascii=False)

    print(f"Saved combined cls_train.json to {output_labels_path}")


if __name__ == "__main__":
    merge_datasets() 