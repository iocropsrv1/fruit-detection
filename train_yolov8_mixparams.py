#!/usr/bin/env python3
"""
YOLOv8를 활용한 과실 Detection 모델 학습 (+ 강화된 Augmentations)
- 내장: mosaic, mixup, copy_paste, HSV jitter
- 커스텀: CLAHE, gamma/contrast 강화 (콜백으로 배치 전처리에서 적용)
"""

import os
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import argparse
import logging
from datetime import datetime
import shutil
import random
import numpy as np
import cv2

# =========================
# 로깅
# =========================
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

# =========================
# GPU 체크
# =========================
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

# =========================
# data.yaml 검증
# =========================
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

# =========================
# 커스텀 콜백: CLAHE + Gamma/Contrast
# =========================
def _apply_clahe_gamma_contrast(img_np_uint8, use_clahe=True, clahe_clip=2.0, clahe_grid=8,
                                gamma_range=(0.8, 1.2), contrast_range=(0.9, 1.1),
                                p_clahe=0.7, p_gamma=0.8, p_contrast=0.8, rng=None):
    """img_np_uint8: BGR(H,W,3), np.uint8"""
    if rng is None:
        rng = random

    out = img_np_uint8

    # CLAHE (V 채널에 적용)
    if use_clahe and rng.random() < p_clahe:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(clahe_grid, clahe_grid))
        v = clahe.apply(v)
        hsv[:, :, 2] = v
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Gamma
    if rng.random() < p_gamma:
        g = rng.uniform(gamma_range[0], gamma_range[1])
        # gamma 보정: out = 255 * (img/255) ** g
        table = np.array([((i / 255.0) ** g) * 255.0 for i in range(256)]).astype(np.uint8)
        out = cv2.LUT(out, table)

    # Contrast (간단한 gain, center 128 기준)
    if rng.random() < p_contrast:
        c = rng.uniform(contrast_range[0], contrast_range[1])
        out = cv2.addWeighted(out, c, np.zeros_like(out), 0, 128*(1-c))

    return out

def build_preprocess_callback(
    enable_clahe_gamma=True,
    clahe_clip=2.0,
    clahe_grid=8,
    gamma_low=0.8,
    gamma_high=1.2,
    contrast_low=0.9,
    contrast_high=1.1,
    p_clahe=0.7,
    p_gamma=0.8,
    p_contrast=0.8,
    seed=42
):
    rng = random.Random(seed)

    def on_preprocess_batch(trainer):
        # trainer.batch: dict with 'img' (torch.FloatTensor, shape [B,3,H,W], 0-1)
        batch = trainer.batch
        if batch is None or 'img' not in batch:
            return
        if not enable_clahe_gamma:
            return

        imgs = batch['img']  # torch.FloatTensor
        if not isinstance(imgs, torch.Tensor):
            return

        imgs_np = imgs.detach().cpu().numpy()  # [B,3,H,W], 0~1 float
        B, C, H, W = imgs_np.shape
        for i in range(B):
            # to uint8 BGR
            img_chw = imgs_np[i]
            img_hwc = np.transpose(img_chw, (1, 2, 0))  # HWC, RGB
            img_u8 = np.clip(img_hwc * 255.0, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

            # apply
            img_bgr = _apply_clahe_gamma_contrast(
                img_bgr,
                use_clahe=True,
                clahe_clip=clahe_clip,
                clahe_grid=clahe_grid,
                gamma_range=(gamma_low, gamma_high),
                contrast_range=(contrast_low, contrast_high),
                p_clahe=p_clahe, p_gamma=p_gamma, p_contrast=p_contrast,
                rng=rng
            )

            # back to tensor [0,1], RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb.astype(np.float32) / 255.0
            imgs_np[i] = np.transpose(img_rgb, (2, 0, 1))

        # 덮어쓰기
        batch['img'] = torch.from_numpy(imgs_np).to(imgs.device)

    return on_preprocess_batch

# =========================
# 학습 설정
# =========================
def create_training_config(
    model_size='s',
    epochs=10,
    batch_size=32,
    imgsz=1080,
    optimizer='SGD',
    # 내장 Augs (요청 반영)
    mosaic=1.0,          # 0.0~1.0 (확률/강도)
    mixup=0.7,           # 이미 인자로 있던 값 유지/노출
    copy_paste=0.5,      # 0~1
    hsv_h=0.015,         # hue jitter
    hsv_s=0.7,           # saturation jitter (채도)
    hsv_v=0.7,           # value jitter (명도)
    fliplr=0.5,
    flipud=0.7,
    degrees=90.0,
    translate=0.1,
    scale=0.5,
    shear=5.0,
    perspective=0.0005,
    # 옵티마이저 & loss
    lr0=1E-2,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box_gain=7.5,
    cls_gain=0.5,
    dfl_gain=1.5,
    save_period=-1,
    workers=8,
    **kwargs
):
    config = {
        'model': f'yolov8{model_size}.pt',
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'device': 'auto',
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

        # ===== 내장 데이터 증강 (YOLOv8 하이퍼파라미터) =====
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

        'workers': workers,
        'save': True,
        'save_period': save_period,
        'val': True,
        'verbose': True,
        'seed': 42,
        'exist_ok': True,
    }
    config.update(kwargs)
    return config

# =========================
# 학습 실행
# =========================
def train_yolov8(data_yaml_path, output_dir, config, fruit_name=None,
                 enable_clahe_gamma=True,
                 clahe_clip=2.0, clahe_grid=8,
                 gamma_low=0.8, gamma_high=1.2,
                 contrast_low=0.9, contrast_high=1.1,
                 p_clahe=0.7, p_gamma=0.8, p_contrast=0.8,
                 seed=42):
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fruit_suffix = f"_{fruit_name}" if fruit_name else ""
    run_name = f"train_{timestamp}{fruit_suffix}"
    run_dir = output_dir / run_name

    logger.info(f"학습 시작: {run_name}")
    logger.info(f"출력 경로: {run_dir}")

    try:
        model_name = config['model']
        logger.info(f"모델 로드: {model_name}")
        model = YOLO(model_name)

        logger.info("모델 구조:")
        logger.info(f"  - Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        logger.info(f"  - Trainable params: {sum(p.numel() for p in model.model.parameters() if p.requires_grad):,}")

        # 설정 저장
        config_filename = f"training_config{fruit_suffix}.yaml"
        config_save_path = run_dir / config_filename
        run_dir.mkdir(parents=True, exist_ok=True)
        config_to_save = config.copy()
        if fruit_name:
            config_to_save['fruit_name'] = fruit_name
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"학습 설정 저장: {config_save_path}")

        # ===== 커스텀 콜백 등록 (CLAHE + gamma/contrast) =====
        if enable_clahe_gamma:
            cb = build_preprocess_callback(
                enable_clahe_gamma=True,
                clahe_clip=clahe_clip, clahe_grid=clahe_grid,
                gamma_low=gamma_low, gamma_high=gamma_high,
                contrast_low=contrast_low, contrast_high=contrast_high,
                p_clahe=p_clahe, p_gamma=p_gamma, p_contrast=p_contrast,
                seed=seed
            )
            # YOLOv8 콜백 훅: on_preprocess_batch
            model.add_callback("on_preprocess_batch", cb)
            logger.info("커스텀 Augmentation 콜백 등록: CLAHE + gamma/contrast")

        # ===== 학습 =====
        logger.info("학습 실행 중...")
        results = model.train(
            data=data_yaml_path,
            project=str(output_dir),
            name=run_name,
            **{k: v for k, v in config.items() if k != 'model'}
        )
        logger.info("✅ 학습 완료!")

        # 결과 요약
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
            if fruit_name:
                results_copy = run_dir / f'results_{fruit_name}.png'
                shutil.copy2(results_png, results_copy)
                logger.info(f"과실명 포함 결과 그래프: {results_copy}")

        confusion_matrix_png = run_dir / 'confusion_matrix.png'
        if confusion_matrix_png.exists() and fruit_name:
            confusion_copy = run_dir / f'confusion_matrix_{fruit_name}.png'
            shutil.copy2(confusion_matrix_png, confusion_copy)
            logger.info(f"과실명 포함 Confusion Matrix: {confusion_copy}")

        return run_dir

    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        raise

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description='YOLOv8 과실 성숙도 검출 모델 학습 (+강화 Augmentations)')

    # 필수
    parser.add_argument('--data', required=True, help='data.yaml 파일 경로')
    parser.add_argument('--output_dir', required=True, help='학습 결과 저장 경로')

    # 선택
    parser.add_argument('--fruit', type=str, help='과실명(선택) - 결과 파일명에 추가')
    parser.add_argument('--model_size', default='s', choices=['n','s','m','l','x'], help='모델 크기')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--imgsz', type=int, default=1080)
    parser.add_argument('--lr0', type=float, default=1e-2)
    parser.add_argument('--device', default='auto', help='디바이스 (auto, 0, 1, cpu)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    # ===== 내장 Augmentation 파라미터 =====
    parser.add_argument('--mosaic', type=float, default=1.0)
    parser.add_argument('--mixup', type=float, default=0.7)
    parser.add_argument('--copy_paste', type=float, default=0.5)
    parser.add_argument('--hsv_h', type=float, default=0.015)
    parser.add_argument('--hsv_s', type=float, default=0.7)
    parser.add_argument('--hsv_v', type=float, default=0.7)
    parser.add_argument('--fliplr', type=float, default=0.5)
    parser.add_argument('--flipud', type=float, default=0.7)
    parser.add_argument('--degrees', type=float, default=90.0)
    parser.add_argument('--translate', type=float, default=0.1)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--shear', type=float, default=5.0)
    parser.add_argument('--perspective', type=float, default=0.0005)

    # ===== 커스텀(CLAHE + gamma/contrast) 파라미터 =====
    parser.add_argument('--enable_clahe_gamma', action='store_true', help='CLAHE & gamma/contrast 콜백 활성화')
    parser.add_argument('--clahe_clip', type=float, default=2.0)
    parser.add_argument('--clahe_grid', type=int, default=8)
    parser.add_argument('--gamma_low', type=float, default=0.8)
    parser.add_argument('--gamma_high', type=float, default=1.2)
    parser.add_argument('--contrast_low', type=float, default=0.9)
    parser.add_argument('--contrast_high', type=float, default=1.1)
    parser.add_argument('--p_clahe', type=float, default=0.7)
    parser.add_argument('--p_gamma', type=float, default=0.8)
    parser.add_argument('--p_contrast', type=float, default=0.8)

    args = parser.parse_args()

    if args.device not in ("auto", "cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    logger = setup_logging(args.output_dir, args.fruit)

    try:
        _ = verify_gpu_availability()
        _ = load_and_verify_config(args.data)

        training_config = create_training_config(
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            lr0=args.lr0,
            workers=args.workers,
            device=args.device,
            seed=args.seed,
            # 내장 augs
            mosaic=args.mosaic,
            mixup=args.mixup,
            copy_paste=args.copy_paste,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            fliplr=args.fliplr,
            flipud=args.flipud,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            shear=args.shear,
            perspective=args.perspective,
        )

        logger.info("=== 학습 설정 요약 ===")
        for k, v in training_config.items():
            logger.info(f"{k}: {v}")

        run_dir = train_yolov8(
            args.data, args.output_dir, training_config, args.fruit,
            enable_clahe_gamma=args.enable_clahe_gamma,
            clahe_clip=args.clahe_clip, clahe_grid=args.clahe_grid,
            gamma_low=args.gamma_low, gamma_high=args.gamma_high,
            contrast_low=args.contrast_low, contrast_high=args.contrast_high,
            p_clahe=args.p_clahe, p_gamma=args.p_gamma, p_contrast=args.p_contrast,
            seed=args.seed
        )

        success_msg = f"🎉 학습 완료! 결과는 다음 경로에 저장되었습니다: {run_dir}"
        if args.fruit:
            success_msg += f"\n🍎 과실: {args.fruit}"
        logger.info(success_msg)

    except Exception as e:
        logger.error(f"💥 학습 실패: {e}")
        raise

if __name__ == "__main__":
    main()
