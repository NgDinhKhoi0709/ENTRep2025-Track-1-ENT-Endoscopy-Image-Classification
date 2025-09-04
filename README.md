# ENTRep 2025 — Track 1: ENT Endoscopy Image Classification

This repository contains our solution for Track 1 (Image Classification) of the ENTRep Challenge at ACM MM 2025. The task is to classify ENT endoscopy images by anatomical region and pathology.

Challenge page: <https://aichallenge.hcmus.edu.vn/acm-mm-2025/entrep>

## Approach

- We use the Surgical Vision-Language Pretraining project (SurgVLP) with the `config_peskavlp` configuration (PeskaVLP) as an image encoder.
- The encoder is frozen and a lightweight multi-layer classification head is added for 7 ENT classes:
  - `nose-right`, `nose-left`, `ear-right`, `ear-left`, `vc-open`, `vc-closed`, `throat`.
- We fine-tune only the classification head on our ENT data. Training uses Focal Loss and tracks both Accuracy and Balanced Accuracy; best checkpoints are saved by each metric.

Backbone reference: <https://github.com/CAMMA-public/SurgVLP.git>

## Repository layout (key files)

- `utils/make_cls_json.py`: Convert `data.json` into `cls.json` mapping `Path -> Classification`.
- `utils/augment_dataset.py`: Create augmented images and labels.
- `utils/merge_train_and_aug.py`: Merge original and augmented label mappings into a combined mapping for training.
- `utils/finetune.py`: Load PeskaVLP (`config_peskavlp`), attach the classification head, and train (optional local training).
- `utils/peskavlp.ipynb`: Kaggle notebook for training with easy environment setup.
- `utils/predict.py`: Local inference script to generate predictions from trained weights.

## Environment setup

### Local (Windows example)

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn pandas tqdm opencv-python pillow mmengine

# Clone SurgVLP (required) and install it locally
git clone https://github.com/CAMMA-public/SurgVLP.git ./SurgVLP
pip install -r SurgVLP/requirements.txt
pip install -e ./SurgVLP
pip install git+https://github.com/openai/CLIP.git
```

Place the PeskaVLP checkpoint at `SurgVLP/weights/PeskaVLP.pth`.

### Kaggle (as used in `utils/peskavlp.ipynb`)

Add cells with:

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/CAMMA-public/SurgVLP.git
```

## Data preparation (recommended: Kaggle merged dataset)

- Kaggle dataset (already merged original + augmented): <https://www.kaggle.com/datasets/ngdihkhoi/augmented-data>
- You can SKIP all local build/augment/merge steps.
- Place the provided merged `cls_train.json` and the images directly under `dataset/augmented_merge_original/` to match `utils/finetune.py` expectations:
  - `dataset/augmented_merge_original/cls_train.json`
  - `dataset/augmented_merge_original/images/` (all images)

## Training

### On Kaggle

- Open `utils/peskavlp.ipynb`, run the installation cells as above, and train. The notebook uses the PeskaVLP backbone and fine-tunes the classification head on ENT data.

### Locally (optional)

- `utils/finetune.py` loads PeskaVLP via `SurgVLP/tests/config_peskavlp.py` and expects a checkpoint at `SurgVLP/weights/PeskaVLP.pth`.

```bash
python utils/finetune.py
```

Checkpoints are saved under `./weights/`:

- `best_balacc.pth` — best balanced accuracy
- `best_acc.pth` — best overall accuracy
- `last.pth` — last epoch

## Inference (local)

Use `utils/predict.py` to run predictions with a trained checkpoint:

```bash
python utils/predict.py --weights ./weights/all_data_fl.pth \
  --output_json ./results/predictions_all_data_fl.json \
  --csv_path ./Dataset/PublicTest/cls.csv \
  --image_dir ./Dataset/PublicTest/PublicTest

# or to scan all images under image_dir (ignoring CSV):
python utils/predict.py --weights ./weights/all_data_fl.pth \
  --output_json ./results/predictions_all_data_fl.json \
  --image_dir ./Dataset/PublicTest/PublicTest --test
```

## Downloads

- Original dataset (Track 1): <https://aichallenge.hcmus.edu.vn/acm-mm-2025/entrep>
- Prebuilt merged augmented dataset (Kaggle): <https://www.kaggle.com/datasets/ngdihkhoi/augmented-data>
- Trained weights: <https://drive.google.com/file/d/1IzhFz7lAtepLu8b_pqZ19VdwfXd3V19H/view?usp=sharing>

## Notes

- The encoder is frozen by default; only the classification head is trained. You can change `freeze_encoder` in `EndoscopeClassifier` to unfreeze.
- Input transforms use the `preprocess` returned by `surgvlp.load()` to stay consistent with the backbone.

## References

- ENTRep Challenge: <https://aichallenge.hcmus.edu.vn/acm-mm-2025/entrep>
- SurgVLP: <https://github.com/CAMMA-public/SurgVLP.git>

