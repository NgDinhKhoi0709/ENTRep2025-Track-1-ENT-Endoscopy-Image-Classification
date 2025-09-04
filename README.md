# ENTRep 2025 — Track 1: Phân loại ảnh nội soi ENT

Kho lưu trữ này là lời giải của chúng tôi cho Track 1 (Image Classification) thuộc cuộc thi ENTRep tại ACM MM 2025. Nhiệm vụ: phân loại ảnh nội soi tai–mũi–họng (ENT) theo vùng giải phẫu và bệnh lý.

Trang cuộc thi: [ENTRep Challenge — Track 1](https://aichallenge.hcmus.edu.vn/acm-mm-2025/entrep)

## Cách tiếp cận

- Sử dụng dự án Surgical Vision-Language Pretraining (SurgVLP) làm backbone hình ảnh, cụ thể cấu hình `config_peskavlp` (PeskaVLP).
- Đóng băng (freeze) encoder của PeskaVLP và thêm một classification head nhiều lớp nhẹ để học 7 lớp ENT:
  - `nose-right`, `nose-left`, `ear-right`, `ear-left`, `vc-open`, `vc-closed`, `throat`.
- Fine-tune chỉ classification head trên dữ liệu ENT của mình. Huấn luyện dùng Focal Loss, theo dõi Accuracy và Balanced Accuracy; lưu checkpoint tốt nhất theo từng chỉ số.

Tham chiếu backbone: [CAMMA-public/SurgVLP](https://github.com/CAMMA-public/SurgVLP.git)

## Cấu trúc repo (các file chính)

- `utils/make_cls_json.py`: Chuyển `data.json` (ban tổ chức) thành `cls.json` dạng ánh xạ `Path -> Classification`.
- `utils/augment_dataset.py`: Tạo ảnh tăng cường (augmentation) và nhãn tương ứng.
- `utils/merge_train_and_aug.py`: Gộp nhãn gốc và nhãn tăng cường, viết ra ánh xạ kết hợp để huấn luyện.
- `utils/finetune.py`: Nạp PeskaVLP (config `config_peskavlp`), gắn classification head và huấn luyện trên tập kết hợp.

## Chuẩn bị môi trường

1) Clone SurgVLP vào thư mục gốc dự án này:

```bash
git clone https://github.com/CAMMA-public/SurgVLP.git ./SurgVLP
```

2) Thiết lập Python (ví dụ trên Windows CMD):

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install -r SurgVLP/requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn pandas tqdm opencv-python pillow mmengine
```

3) Trọng số: đặt checkpoint PeskaVLP tại `SurgVLP/weights/PeskaVLP.pth`.

## Chuẩn bị dữ liệu

Kỳ vọng cấu trúc dữ liệu gốc của Track 1:

```
dataset/
  train/
    images/            # ảnh gốc
    data.json          # annotation từ BTC
```

1) Tạo `cls_train.json` (ánh xạ `Path -> Classification`) từ `data.json`:

```bash
python utils/make_cls_json.py --input dataset/train/data.json --output dataset/train/cls_train.json
```

2) Tăng cường dữ liệu. Mặc định sẽ ghi ảnh vào `dataset/augmented/` và nhãn vào `dataset/augmentation/cls_augmented.json`:

```bash
python utils/augment_dataset.py
```

Chuyển file nhãn tăng cường về đường dẫn mà bước gộp mong đợi:

```bash
mkdir dataset\augmented 2>NUL
move dataset\augmentation\cls_augmented.json dataset\augmented\cls_augmented.json
```

Tuỳ chọn (tải sẵn augmented data):

- Bạn có thể tải bộ dữ liệu đã tăng cường sẵn tại Kaggle: <https://www.kaggle.com/datasets/ngdihkhoi/augmented-data>. Khi đó, giải nén vào `dataset/augmented/` và đảm bảo có file nhãn `dataset/augmented/cls_augmented.json`, rồi chuyển sang bước gộp (bước 3).

3) Gộp nhãn gốc và nhãn tăng cường, xuất ánh xạ kết hợp:

```bash
python utils/merge_train_and_aug.py
```

Đầu ra:

```
dataset/
  augmented_merge_original/
    cls_train.json
```

4) Chuẩn bị thư mục ảnh cho huấn luyện như `utils/finetune.py` mong đợi.

`utils/finetune.py` đọc ảnh ở `dataset/augmented_merge_original/images`. Sao chép toàn bộ ảnh từ `dataset/augmented/` sang đó (bao gồm ảnh gốc đã copy vào `augmented` ở bước gộp):

```bash
mkdir dataset\augmented_merge_original\images 2>NUL
robocopy dataset\augmented dataset\augmented_merge_original\images /E
```

## Tải dữ liệu/weights

- Dữ liệu gốc (Track 1): <https://aichallenge.hcmus.edu.vn/acm-mm-2025/entrep>
- Dữ liệu augmented sẵn: <https://www.kaggle.com/datasets/ngdihkhoi/augmented-data>
- Trọng số sau khi huấn luyện (weights): <https://drive.google.com/file/d/1IzhFz7lAtepLu8b_pqZ19VdwfXd3V19H/view?usp=sharing>

## Huấn luyện (fine-tune)

`utils/finetune.py` nạp PeskaVLP thông qua `SurgVLP/tests/config_peskavlp.py` và checkpoint tại `SurgVLP/weights/PeskaVLP.pth`.

Chạy huấn luyện:

```bash
python utils/finetune.py
```

Checkpoint sẽ được lưu ở `./weights/`:

- `best_balacc.pth` — balanced accuracy tốt nhất (trên toàn bộ tập train)
- `best_acc.pth` — accuracy tốt nhất (trên toàn bộ tập train)
- `last.pth` — epoch cuối

Ghi chú:

- Mặc định encoder bị đóng băng; chỉ classification head được học. Có thể điều chỉnh `freeze_encoder` trong `EndoscopeClassifier` nếu muốn unfreeze.
- Tiền xử lý ảnh dùng `preprocess` trả về từ `surgvlp.load()` để nhất quán với backbone.

## Trích dẫn/Tham khảo

- ENTRep Challenge: <https://aichallenge.hcmus.edu.vn/acm-mm-2025/entrep>
- SurgVLP: <https://github.com/CAMMA-public/SurgVLP.git>

