# OCR-TTA (Train & ONNX Export)

TPS-ResNet-BiLSTM-Attn 파이프라인으로 텍스트 인식 모델을 학습하고, 검증 시점마다 **ONNX**로 자동 내보내는 스크립트와 최소 구성입니다.

---

## 📂 프로젝트 구조

```
ocr_tta/
├── data_lmdb_08/                 # LMDB 데이터 루트 (train/val 포함)
├── dataset.py                    # 데이터셋/Collate 유틸
├── final_charlist_1770.txt       # 문자셋 파일(줄당 1문자)
├── model.py                      # 모델 정의 (TPS/Backbone/Sequence/Prediction)
├── run.sh                        # 실행 스크립트(예시)
├── test.py                       # validation() 구현
├── train9_onnx.py                # 학습 + ONNX export 메인 스크립트
└── utils.py                      # 라벨 컨버터/유틸
```

---

## ⚙️ 요구 사항

- Python 3.8+  
- PyTorch & torchvision (CUDA 권장)  
- lmdb, pillow, numpy, tqdm  
- onnx, onnxruntime  
- (옵션) BaiduWarpCTC (`--baiduCTC` 쓸 때만)

설치:
```bash
pip install torch torchvision lmdb pillow numpy tqdm onnx onnxruntime
```

---

## 🔤 문자셋

- `final_charlist_1770.txt` → **한 줄당 1문자**  
- 학습 시작 시: 파일 문자셋 + LMDB에서 추출한 문자들의 **합집합**을 최종 문자셋으로 사용

---

## 🚀 실행 방법

### 1) `run.sh` 사용
```bash
bash ./run.sh
```

### 2) 직접 실행
```bash
python3 -W ignore train9_onnx.py   --exp_name        .   --train_data      /home/kgh/ocr-tta_copy/data_lmdb_08/train/custom   --valid_data      /home/kgh/ocr-tta_copy/data_lmdb_08/val   --select_data     custom   --batch_ratio     2.5   --Transformation  TPS   --FeatureExtraction ResNet   --SequenceModeling BiLSTM   --Prediction      Attn   --character       /home/kgh/ocr-tta/final_charlist_1770.txt   --num_iter        12000   --valInterval     200   --data_filtering_off   --save_path       ./saved_models_set_onnx/saved_models_9_all_generate_hub_08.06
```

---

## 📊 학습 로직 개요

- `valInterval`마다 **검증** 수행  
- 최고 성능 갱신 시:  
  - `best_accuracy.pth / .onnx`  
  - `best_norm_ED.pth / .onnx`  
- 개선 없으면 **Early Stopping**  
- 마지막 이터레이션: `final.pth / .onnx` 저장  
- ONNX: opset 16 + `adaptive_avg_pool2d` 심볼릭 패치 적용

---

## 📑 파일별 설명 (요약)

- **train9_onnx.py** — 학습/검증 드라이버. 주기적 평가, 최고 성능 시 `.pth`/`.onnx` 저장, 문자셋 자동 보강.  
- **dataset.py** — LMDB 로더·배치 밸런싱·전처리(`AlignCollate`).  
- **model.py** — TPS(선택) → FeatureExtraction(VGG/RCNN/ResNet) → BiLSTM(선택) → Prediction(CTC/Attn) 구조.  
- **utils.py** — 라벨 컨버터(CTC/Attn), 평균기 등 유틸.  
- **test.py** — `validation(...)` 구현(손실/정확도/NED/추론시간).  
- **run.sh** — 실행 예시 스크립트(GPU/옵션).  
- **data_lmdb_08/** — LMDB 루트(`train/custom`, `val`에 `data.mdb/lock.mdb`).  
- **final_charlist_1770.txt** — 기본 문자셋(줄당 1문자). 학습 시 LMDB 문자와 합쳐 사용.


---

