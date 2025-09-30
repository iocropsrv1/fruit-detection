#!/usr/bin/env python3
"""
YOLOv8 과실 Detection 모델 학습 (Detection 유효 인자만)
- 내장 증강: mosaic, mixup, copy_paste, hsv_h/s/v, fliplr/flipud,
             degrees/translate/scale/shear/perspective
- 기타: close_mosaic, freeze, cos_lr 등
- 직관적인 디렉토리명으로 결과 저장
- 샘플 이미지 추적 기능: train 원본 3장 + 각 단계 결과 저장
- 수정: 클래스별 색상 구분 bbox만 표시 (텍스트 제거)
"""

import os
import yaml
import torch
import random
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
import logging
from datetime import datetime
import shutil
from PIL import Image, ImageDraw, ImageFont

def setup_logging(log_dir, fruit_name=None):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fruit_suffix = f"_{fruit_name}" if fruit_name else ""
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}{fruit_suffix}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    if fruit_name:
        logger.info(f"🍎 과실 종류: {fruit_name}")
    return logger

def verify_gpu_availability():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"✅ GPU 사용 가능: {gpu_count}개 GPU 감지")
        print(f"현재 GPU: {gpu_name}")
        print(f"CUDA 버전: {torch.version.cuda}")
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        print(f"GPU 메모리: {total_memory / 1024**3:.1f}GB")
        return True
    else:
        print("⚠️  GPU를 사용할 수 없습니다. CPU로 학습하면 시간이 오래 걸립니다.")
        return False

def load_and_verify_config(data_yaml_path):
    data_yaml_path = Path(data_yaml_path)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml 파일을 찾을 수 없습니다: {data_yaml_path}")
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    required_keys = ['path', 'train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"data.yaml에 필수 키가 없습니다: {key}")
    base_path = Path(config['path'])
    for split in ['train', 'val']:
        if split in config:
            split_path = base_path / config[split]
            if not split_path.exists():
                raise FileNotFoundError(f"경로가 존재하지 않습니다: {split_path}")
    if len(config['names']) != config['nc']:
        raise ValueError(f"클래스 수 불일치: nc={config['nc']}, names 길이={len(config['names'])}")
    print("✅ 데이터 설정 검증 완료")
    print(f"   - 클래스 수: {config['nc']}")
    print(f"   - 클래스: {config['names']}")
    print(f"   - 데이터 경로: {config['path']}")
    return config

def select_sample_images(data_config, num_samples=3):
    """train 폴더에서 랜덤하게 샘플 이미지들을 선택"""
    logger = logging.getLogger(__name__)
    base_path = Path(data_config['path'])
    train_images_path = base_path / data_config['train'] / 'images'
    if not train_images_path.exists():
        train_images_path = base_path / data_config['train']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(train_images_path.glob(f'*{ext}')))
        all_images.extend(list(train_images_path.glob(f'*{ext.upper()}')))
    if len(all_images) < num_samples:
        logger.warning(f"train 폴더에 이미지가 {len(all_images)}장 밖에 없어서 모두 선택합니다.")
        num_samples = len(all_images)
    selected_images = random.sample(all_images, num_samples) if all_images else []
    logger.info(f"📸 샘플 이미지 {len(selected_images)}장 선택:")
    for i, img_path in enumerate(selected_images, 1):
        logger.info(f"   {i}. {img_path.name}")
    return selected_images

def save_sample_original_images(selected_images, run_dir):
    """선택된 원본 이미지들을 samples 폴더에 저장"""
    logger = logging.getLogger(__name__)
    samples_dir = run_dir / 'samples'
    original_dir = samples_dir / '01_original'
    original_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for i, img_path in enumerate(selected_images, 1):
        dest_path = original_dir / f"sample_{i:02d}_{img_path.name}"
        shutil.copy2(img_path, dest_path)
        saved_paths.append(dest_path)
        logger.info(f"   원본 저장: {dest_path.name}")
    return saved_paths

def create_augmented_samples(selected_images, run_dir, config):
    """선택된 이미지들에 대해 augmentation 예시 생성"""
    logger = logging.getLogger(__name__)
    samples_dir = run_dir / 'samples'
    aug_dir = samples_dir / '02_augmented'
    aug_dir.mkdir(parents=True, exist_ok=True)
    try:
        # 간단한 전처리(리사이즈+패딩)만 예시로 저장
        for i, img_path in enumerate(selected_images, 1):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                height, width = img.shape[:2]
                target_size = config['imgsz']
                scale = min(target_size/width, target_size/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                resized_img = cv2.resize(img, (new_width, new_height))
                pad_width = (target_size - new_width) // 2
                pad_height = (target_size - new_height) // 2
                padded_img = cv2.copyMakeBorder(
                    resized_img, pad_height, target_size-new_height-pad_height,
                    pad_width, target_size-new_width-pad_width,
                    cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )
                dest_path = aug_dir / f"sample_{i:02d}_preprocessed.jpg"
                cv2.imwrite(str(dest_path), padded_img)
            except Exception as e:
                logger.warning(f"샘플 {i} 전처리 중 오류: {e}")
                continue
        logger.info(f"📸 전처리된 샘플 이미지들이 저장되었습니다: {aug_dir}")
    except Exception as e:
        logger.warning(f"Augmentation 샘플 생성 중 오류: {e}")

def get_class_color(class_id):
    """클래스 ID에 따른 색상 반환 (BGR 형식)"""
    colors = {
        0: (0, 0, 255),    # ripened - 빨간색
        1: (0, 255, 0),    # ripening - 연두색
        2: (255, 0, 0),    # unripened - 파란색
    }
    return colors.get(class_id, (128, 128, 128))  # 기본값: 회색

def create_validation_predictions(selected_images, model, run_dir, data_config):
    """선택된 이미지들에 대한 validation 예측 결과 생성 - 클래스별 색상 bbox만"""
    logger = logging.getLogger(__name__)
    samples_dir = run_dir / 'samples'
    val_dir = samples_dir / '03_validation'
    val_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        for i, img_path in enumerate(selected_images, 1):
            try:
                results = model(str(img_path), verbose=False)
                result = results[0]
                
                # 원본 이미지 로드
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # bbox 그리기 (텍스트 없이, 클래스별 색상)
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # 클래스별 색상 가져오기
                        color = get_class_color(cls)
                        
                        # bbox만 그리기 (텍스트 제거)
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                dest_path = val_dir / f"sample_{i:02d}_validation.jpg"
                cv2.imwrite(str(dest_path), img)
                
            except Exception as e:
                logger.warning(f"샘플 {i} validation 예측 중 오류: {e}")
                continue
                
        logger.info(f"📸 Validation 예측 결과가 저장되었습니다: {val_dir}")
        logger.info("   색상 구분: 빨강(ripened), 연두(ripening), 파랑(unripened)")
        
    except Exception as e:
        logger.warning(f"Validation 예측 생성 중 오류: {e}")

def create_test_predictions(model, run_dir, data_config, num_samples=3):
    """테스트용 랜덤 예측 결과 생성 - 클래스별 색상 bbox만"""
    logger = logging.getLogger(__name__)
    samples_dir = run_dir / 'samples'
    test_dir = samples_dir / '04_test_results'
    test_dir.mkdir(parents=True, exist_ok=True)
    
    base_path = Path(data_config['path'])
    val_images_path = base_path / data_config['val'] / 'images'
    if not val_images_path.exists():
        val_images_path = base_path / data_config['val']
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(val_images_path.glob(f'*{ext}')))
        all_images.extend(list(val_images_path.glob(f'*{ext.upper()}')))
    
    if len(all_images) == 0:
        logger.warning("테스트할 validation 이미지를 찾을 수 없습니다.")
        return
    
    test_samples = min(num_samples, len(all_images))
    selected_test_images = random.sample(all_images, test_samples)
    
    try:
        for i, img_path in enumerate(selected_test_images, 1):
            try:
                results = model(str(img_path), verbose=False)
                result = results[0]
                
                # 원본 이미지 로드
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # bbox 그리기 (텍스트 없이, 클래스별 색상)
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # 클래스별 색상 가져오기
                        color = get_class_color(cls)
                        
                        # bbox만 그리기 (텍스트 제거)
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                dest_path = test_dir / f"test_{i:02d}_{img_path.name}"
                cv2.imwrite(str(dest_path), img)
                
            except Exception as e:
                logger.warning(f"테스트 샘플 {i} 예측 중 오류: {e}")
                continue
                
        logger.info(f"📸 테스트 예측 결과가 저장되었습니다: {test_dir}")
        logger.info("   색상 구분: 빨강(ripened), 연두(ripening), 파랑(unripened)")
        
    except Exception as e:
        logger.warning(f"테스트 예측 생성 중 오류: {e}")

def create_sample_summary(run_dir, selected_images):
    """샘플 이미지들에 대한 요약 정보 생성"""
    logger = logging.getLogger(__name__)
    samples_dir = run_dir / 'samples'
    summary_file = samples_dir / 'sample_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== 샘플 이미지 추적 결과 ===\n\n")
        f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("선택된 원본 이미지들:\n")
        for i, img_path in enumerate(selected_images, 1):
            f.write(f"{i}. {img_path.name}\n")
        f.write("\n폴더 구조:\n")
        f.write("├── 01_original/     - 선택된 원본 이미지들\n")
        f.write("├── 02_augmented/    - 전처리된 이미지들\n")
        f.write("├── 03_validation/   - 학습된 모델의 예측 결과\n")
        f.write("├── 04_test_results/ - 테스트 이미지 예측 결과\n")
        f.write("└── sample_summary.txt - 이 파일\n\n")
        f.write("클래스별 색상 구분:\n")
        f.write("- 0 (ripened): 빨간색\n")
        f.write("- 1 (ripening): 연두색\n")
        f.write("- 2 (unripened): 파란색\n")
    logger.info(f"📄 샘플 요약 정보 저장: {summary_file}")

def create_training_config(
    model_size='s',
    epochs=10,
    batch_size=32,
    imgsz=1080,
    optimizer='SGD',
    # ---- Detection에서 유효한 Augmentations ----
    mosaic=1.0,
    mixup=0.7,
    copy_paste=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    flipud=0.0,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    close_mosaic=10,   # 마지막 N epoch에서 mosaic 끄기
    # ---- Optim/Loss ----
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box_gain=7.5,
    cls_gain=0.5,
    dfl_gain=1.5,
    # ---- 기타 ----
    freeze=0,          # 0이면 동결 없음. 정수면 앞쪽 n 레이어 동결
    cos_lr=False,      # 코사인 LR 스케줄 사용 여부
    save_period=-1,
    workers=8,
    device=None,       # ← 추가: None이면 키를 넣지 않음
    **kwargs
):
    config = {
        'model': f'yolov8{model_size}.pt',
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'optimizer': optimizer,
        'lr0': lr0,
        'lrf': lrf,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'warmup_epochs': warmup_epochs,
        'warmup_momentum': warmup_momentum,
        'warmup_bias_lr': warmup_bias_lr,
        'box': box_gain,
        'cls': cls_gain,
        'dfl': dfl_gain,

        # ---- Detection Augs ----
        'mosaic': mosaic,
        'mixup': mixup,
        'copy_paste': copy_paste,
        'hsv_h': hsv_h,
        'hsv_s': hsv_s,
        'hsv_v': hsv_v,
        'fliplr': fliplr,
        'flipud': flipud,
        'degrees': degrees,
        'translate': translate,
        'scale': scale,
        'shear': shear,
        'perspective': perspective,
        'close_mosaic': close_mosaic,

        # ---- 기타 ----
        'freeze': freeze,
        'cos_lr': cos_lr,

        'workers': workers,
        'save': True,
        'save_period': save_period,
        'val': True,
        'verbose': True,
        'seed': 42,
        'exist_ok': True,
    }
    # device 지정 시에만 키 추가 (Ultralytics 기본 동작 유지)
    if device is not None:
        config['device'] = device
    config.update(kwargs)
    return config

def generate_descriptive_run_name(config, fruit_name=None):
    """직관적인 디렉토리명 생성 - 주요 파라미터만 포함"""
    parts = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    parts.append(f"train_{timestamp}")
    if fruit_name:
        parts.append(f"fruit_{fruit_name}")
    model_name = config['model'].replace('yolov8', '').replace('.pt', '')
    parts.append(f"model_{model_name}")
    parts.append(f"ep{config['epochs']}")
    parts.append(f"bs{config['batch']}")
    parts.append(f"lr{config['lr0']}")
    parts.append(f"opt{config['optimizer']}")
    if config['imgsz'] != 640:
        parts.append(f"img{config['imgsz']}")
    aug_params = []
    if config['mixup'] != 0.0:
        aug_params.append(f"mixup{config['mixup']}")
    if config['mosaic'] != 1.0:
        aug_params.append(f"mosaic{config['mosaic']}")
    if config['flipud'] != 0.0:
        aug_params.append(f"flipud{config['flipud']}")
    if config['fliplr'] != 0.5:
        aug_params.append(f"fliplr{config['fliplr']}")
    if config['degrees'] != 0.0:
        aug_params.append(f"deg{config['degrees']}")
    if config['scale'] != 0.5:
        aug_params.append(f"scale{config['scale']}")
    if config['shear'] != 0.0:
        aug_params.append(f"shear{config['shear']}")
    if config['copy_paste'] != 0.0:
        aug_params.append(f"cp{config['copy_paste']}")
    if config['hsv_h'] != 0.015:
        aug_params.append(f"hsvh{config['hsv_h']}")
    if config['hsv_s'] != 0.7:
        aug_params.append(f"hsvs{config['hsv_s']}")
    if config['hsv_v'] != 0.4:
        aug_params.append(f"hsvv{config['hsv_v']}")
    if aug_params:
        parts.extend(aug_params)
    if config['cos_lr']:
        parts.append("cosLR")
    if config['freeze'] > 0:
        parts.append(f"freeze{config['freeze']}")
    return "_".join(parts)

def train_yolov8(data_yaml_path, output_dir, config, fruit_name=None):
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_config = load_and_verify_config(data_yaml_path)
    run_name = generate_descriptive_run_name(config, fruit_name)
    run_dir = output_dir / run_name
    logger.info(f"학습 시작: {run_name}")
    logger.info(f"출력 경로: {run_dir}")
    try:
        logger.info("🎯 샘플 이미지 선택 중...")
        selected_images = select_sample_images(data_config, num_samples=3)
        save_sample_original_images(selected_images, run_dir)
        logger.info("🔄 전처리 샘플 생성 중...")
        create_augmented_samples(selected_images, run_dir, config)
        model_name = config['model']
        logger.info(f"모델 로드: {model_name}")
        model = YOLO(model_name)
        logger.info("모델 구조:")
        logger.info(f"  - Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        logger.info(f"  - Trainable params: {sum(p.numel() for p in model.model.parameters() if p.requires_grad):,}")
        config_filename = f"training_config.yaml"
        config_save_path = run_dir / config_filename
        run_dir.mkdir(parents=True, exist_ok=True)
        config_to_save = config.copy()
        if fruit_name:
            config_to_save['fruit_name'] = fruit_name
        config_to_save['run_name'] = run_name
        config_to_save['selected_sample_images'] = [str(img) for img in selected_images]
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"학습 설정 저장: {config_save_path}")
        logger.info("🚀 학습 실행 중...")
        results = model.train(
            data=data_yaml_path,
            project=str(output_dir),
            name=run_name,
            **{k: v for k, v in config.items() if k != 'model'}
        )
        logger.info("✅ 학습 완료!")
        logger.info("🎯 Validation 예측 결과 생성 중...")
        best_model_path = run_dir / 'weights' / 'best.pt'
        if best_model_path.exists():
            trained_model = YOLO(str(best_model_path))
            create_validation_predictions(selected_images, trained_model, run_dir, data_config)
            logger.info("🎲 테스트 예측 결과 생성 중...")
            create_test_predictions(trained_model, run_dir, data_config, num_samples=3)
        create_sample_summary(run_dir, selected_images)
        final_model_path = run_dir / 'weights' / 'best.pt'
        if final_model_path.exists():
            logger.info(f"최고 성능 모델: {final_model_path}")
            if fruit_name:
                best_model_copy = run_dir / 'weights' / f'best_{fruit_name}.pt'
                shutil.copy2(final_model_path, best_model_copy)
                logger.info(f"과실명 포함 모델: {best_model_copy}")
        results_png = run_dir / 'results.png'
        if results_png.exists():
            logger.info(f"학습 결과 그래프: {results_png}")
        confusion_matrix_png = run_dir / 'confusion_matrix.png'
        if confusion_matrix_png.exists():
            logger.info(f"Confusion Matrix: {confusion_matrix_png}")
        samples_dir = run_dir / 'samples'
        if samples_dir.exists():
            logger.info(f"📸 샘플 이미지 추적 결과: {samples_dir}")
            logger.info("   ├── 01_original/     - 원본 이미지 3장")
            logger.info("   ├── 02_augmented/    - 전처리된 이미지")
            logger.info("   ├── 03_validation/   - 학습 모델 예측 결과 (클래스별 색상 bbox)")
            logger.info("   ├── 04_test_results/ - 테스트 예측 결과 (클래스별 색상 bbox)")
            logger.info("   └── sample_summary.txt - 요약 정보")
            logger.info("   색상: 빨강(ripened), 연두(ripening), 파랑(unripened)")
        return run_dir
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        raise

def _normalize_device_arg(raw):
    """
    CLI --device 인자를 Ultralytics가 받는 유효 문자열로 정규화.
    - 'cpu' -> 'cpu'
    - 'auto' -> (GPU 있으면 '0', 없으면 'cpu')
    - '0' / '1' / '0,1' 등 -> 그대로
    - 0 / 1 (정수) -> '0', '1'
    """
    if isinstance(raw, int):
        return str(raw)
    s = str(raw).strip()
    if s.lower() == 'cpu':
        return 'cpu'
    if s.lower() == 'auto':
        return '0' if torch.cuda.is_available() else 'cpu'
    # 콤마 구분 멀티-GPU도 그대로 허용
    return s

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 과실 성숙도 검출 모델 학습 (Detection 유효 인자 + 샘플 추적)')
    # 필수
    parser.add_argument('--data', required=True, help='data.yaml 파일 경로')
    parser.add_argument('--output_dir', required=True, help='학습 결과 저장 경로')
    # 선택
    parser.add_argument('--fruit', type=str, help='과실명(선택) - 결과 파일명에 추가')
    parser.add_argument('--model_size', default='s', choices=['n','s','m','l','x'], help='모델 크기')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--imgsz', type=int, default=1080)
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam', 'AdamW'], help='Optimizer')
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--device', default='auto', help="디바이스 ('cpu', 'auto', '0', '0,1' 등)")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    # ---- Detection Augmentations ----
    parser.add_argument('--mosaic', type=float, default=1.0)
    parser.add_argument('--mixup', type=float, default=0.0)
    parser.add_argument('--copy_paste', type=float, default=0.0)
    parser.add_argument('--hsv_h', type=float, default=0.015)
    parser.add_argument('--hsv_s', type=float, default=0.7)
    parser.add_argument('--hsv_v', type=float, default=0.4)
    parser.add_argument('--fliplr', type=float, default=0.5)
    parser.add_argument('--flipud', type=float, default=0.0)
    parser.add_argument('--degrees', type=float, default=0.0)
    parser.add_argument('--translate', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--shear', type=float, default=0.0)
    parser.add_argument('--perspective', type=float, default=0.0)
    parser.add_argument('--close_mosaic', type=int, default=10)
    # ---- 기타 ----
    parser.add_argument('--freeze', type=int, default=0)
    parser.add_argument('--cos_lr', action='store_true', help='Cosine LR 사용')

    args = parser.parse_args()

    # 재현성
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # CUDA_VISIBLE_DEVICES 설정 (선택적): 숫자/목록일 때만 설정
    device_norm = _normalize_device_arg(args.device)
    if device_norm not in ('cpu',) and device_norm.lower() != 'cpu':
        # '0,1' 같은 형식만 환경변수로 반영
        os.environ["CUDA_VISIBLE_DEVICES"] = device_norm

    logger = setup_logging(args.output_dir, args.fruit)

    try:
        _ = verify_gpu_availability()

        training_config = create_training_config(
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            optimizer=args.optimizer,
            lr0=args.lr0,
            workers=args.workers,
            seed=args.seed,
            mosaic=args.mosaic,
            mixup=args.mixup,
            copy_paste=args.copy_paste,
            hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
            fliplr=args.fliplr, flipud=args.flipud,
            degrees=args.degrees, translate=args.translate, scale=args.scale,
            shear=args.shear, perspective=args.perspective,
            close_mosaic=args.close_mosaic,
            freeze=args.freeze,
            cos_lr=args.cos_lr,
            device=device_norm  # ← 여기서 유효한 device 문자열 전달
        )

        logger.info("=== 학습 설정 요약 ===")
        for k, v in training_config.items():
            logger.info(f"{k}: {v}")

        run_dir = train_yolov8(args.data, args.output_dir, training_config, args.fruit)

        success_msg = f"🎉 학습 완료! 결과는 다음 경로에 저장되었습니다: {run_dir}"
        if args.fruit:
            success_msg += f"\n🍎 과실: {args.fruit}"
        success_msg += f"\n📸 샘플 이미지 추적 결과: {run_dir / 'samples'}"
        success_msg += f"\n🎨 클래스별 색상: 빨강(ripened), 연두(ripening), 파랑(unripened)"
        logger.info(success_msg)

    except Exception as e:
        logger.error(f"💥 학습 실패: {e}")
        raise

if __name__ == "__main__":
    main()

