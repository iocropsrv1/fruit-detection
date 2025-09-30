# Fruit Detection - Crop Detection Model Enhancement
헤르마이 예찰 로봇의 작물 인식(Detection) 모델 성능 고도화 프로젝트

## 📋 Overview
본 프로젝트는 YOLOv8 기반의 과일 detection 모델을 학습하고 평가하기 위한 프레임워크입니다. 
다양한 하이퍼파라미터 조합을 자동으로 실험하고, 최적의 모델을 찾기 위함입니다.

### 주요 기능
- 다중 모델, 다중 하이퍼파라미터 조합 자동 학습
- 학습된 모델 일괄 평가 및 결과 비교

## 🔧 Requirements
### System Requirements
- Python ≥ 3.9
- CUDA-capable GPU (recommended)

### Core Dependencies
```txt
ultralytics==8.0.200
torch==2.2.1
torchvision==0.17.1
opencv-python==4.9.0.80
numpy==1.26.4
pandas==2.2.1
pillow==10.2.0
PyYAML==6.0.1
```

### Optional Dependencies
```txt
wandb==0.18.3           # 학습 모니터링
matplotlib==3.8.3       # 시각화
scikit-learn==1.4.1     # 평가 메트릭
gpustat==1.1.1         # GPU 모니터링
```

### Installation
```bash
# 1. 가상환경 생성 (권장)
conda create -n fruit-detection python=3.9
conda activate fruit-detection

# 2. 저장소 클론
git clone https://github.com/iocropsrv1/fruit-detection.git
cd fruit-detection

# 3. 패키지 설치
pip install -r requirements.txt
```

## 📦 Pre-trained Models

학습된 모델 가중치를 다운로드하여 사용할 수 있습니다.

**model_train_datasets**: v1.3 (각 과실별 개별 모델)

**다운로드**: [Google Drive](https://drive.google.com/drive/folders/1V7EV8zabnozVnpCYnLLtT8QExJYOzt5e?usp=drive_link)


## 🚀 Quick Start

### 1. 단일 모델 학습

```bash
python train_yolov8.py \
    --data <dataset_dir> \
    --model_size m \
    --epochs 100 \
    --batch_size 16
```

### 2. 다중 모델 학습
여러 하이퍼파라미터 조합을 자동으로 실험합니다.

```bash
python train_sweeper.py \
    --train_script train_yolov8_mixparams.py \
    --datasets <train_dataset_dir> \
    --models s,m \
    --epochs 50,100 \
    --batch_size 16,32 \
    --optimizer SGD,Adam \
    --lr0 0.01,0.001 \
    --imgsz 640,1080 \
    --output_dir outputs/training_results \
    --max_concurrent 2 \
    --gpus 0,1
```

### 3. 다중 모델 평가

```bash
python eval_sweeper.py \
    --eval_script evaluate_model.py \
    --models outputs/training_results/*/weights/best.pt \
    --datasets <eval_dataset_dir> \
    --fruits pepper,tomato \
    --output_dir outputs/evaluation_results \
    --conf_thresholds 0.25,0.3 \
    --iou_thresholds 0.45,0.5 \
    --samples 20 \
    --max_concurrent 2 \
    --gpus 0,1
```

## 🔍 Dataset Format

### YOLO Format (Required)

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

### data.yaml Example

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 3
names: ['ripened', 'ripening', 'unripened']
```
