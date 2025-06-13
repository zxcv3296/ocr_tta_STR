# ocr_tta_STR
## 1. 데이터 80:20 or 100:0 split
```bash
python split_dataset_train100.py
```

## 2. gt.txt 만들기( 1에서 gt.txt 만들어졌으면 안만들어도 됨)

```bash
python make_full_gt.py \
  --train_dir /mnt/aict_nas/pil_lab/OCR_TTA/korean_plate_data/train \
  --output_gt /mnt/aict_nas/pil_lab/OCR_TTA/korean_plate_data/train_gt.txt
```

/mnt/aict_nas/.../train/인천73아1092.jpg    인천73아1092
/mnt/aict_nas/.../train/인천73아1092-2.jpg 인천73아1092
/mnt/aict_nas/.../train/0100_20240923000157451_전북32바1924_plate.jpg    전북32바1924


## 3. lmdb 생성
```bash
python create_lmdb_dataset.py \
    /mnt/aict_nas/pil_lab/OCR_TTA/korean_plate_data/train_100 \
    /mnt/aict_nas/pil_lab/OCR_TTA/korean_plate_data/train_gt_100.txt \
    /mnt/aict_nas/pil_lab/OCR_TTA/korean_plate_data/data_lmdb/training_100
```

python create_lmdb_dataset.py \
    /mnt/aict_nas/pil_lab/OCR_TTA/korean_plate_data/train_100 \
    /mnt/aict_nas/pil_lab/OCR_TTA/korean_plate_data/train_gt_100.txt \
    /mnt/aict_nas/pil_lab/OCR_TTA/korean_plate_data/data_lmdb/training_100
