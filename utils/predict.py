import json
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

from mmengine.config import Config
import surgvlp

from finetune import EndoscopeClassifier  # reuse the model class

# --------------------------------------------------------------------------------------
# Dataset for PublicTest
# --------------------------------------------------------------------------------------

class PublicTestDataset(Dataset):
    """Dataset for inference over PublicTest image list."""

    def __init__(self, csv_file: str | None, image_root: str, transform):
        """Create dataset.

        If ``csv_file`` is provided, it should be a CSV without header containing
        filenames.  Otherwise, all image files (png/jpg/jpeg) under
        ``image_root`` will be used.
        """

        if csv_file is not None:
            # csv contains filenames, possibly with no header
            self.filenames = pd.read_csv(csv_file, header=None).iloc[:, 0].tolist()
        else:
            exts = {".png", ".jpg", ".jpeg"}
            self.filenames = [
                f.name
                for f in Path(image_root).iterdir()
                if f.suffix.lower() in exts and f.is_file()
            ]
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_root, fname)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, fname

# --------------------------------------------------------------------------------------
# Main inference routine
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Endoscope class prediction script")
    parser.add_argument("--weights", type=str, default="./weights/all_data_fl.pth", help="Path to classifier checkpoint")
    parser.add_argument("--output_json", type=str, default="./results/predictions_all_data_fl.json", help="Output JSON filepath")
    parser.add_argument("--csv_path", type=str, default="./Dataset/PublicTest/cls.csv", help="CSV file listing images (ignored when --test is used)")
    parser.add_argument("--image_dir", type=str, default="./Dataset/PublicTest/PublicTest", help="Directory with images")
    parser.add_argument("--test", action="store_true", help="Predict all images inside --image_dir (default ./Dataset/imgs) and ignore --csv_path")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve paths from arguments
    weight_path = args.weights

    if args.test:
        # Use image_dir directly and ignore CSV
        csv_path = None
    else:
        csv_path = args.csv_path

    image_dir = args.image_dir
    output_json = args.output_json

    # Build base PeskaVLP model
    base_cfg = Config.fromfile("./tests/config_peskavlp.py")["config"]
    peskaVLP_model, preprocess = surgvlp.load(base_cfg["model_config"], device=device, pretrain="./weights/PeskaVLP.pth")

    # Wrap with classifier head
    model = EndoscopeClassifier(peskaVLP_model, num_classes=7, freeze_encoder=True).to(device)

    # Load weights
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # Build dataset & loader
    dataset = PublicTestDataset(csv_path, image_dir, transform=preprocess)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    predictions = {}

    with torch.no_grad():
        for images, fnames in tqdm(loader, desc="Predict"):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            for fname, pred in zip(fnames, preds):
                predictions[fname] = int(pred)

    # Ensure output directory exists
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)

    print(f"Saved predictions to {output_json}. Total images: {len(predictions)}")


if __name__ == "__main__":
    main()
