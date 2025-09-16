#!/usr/bin/env python3
"""
YOLOv8를 활용한 과실 Detection 모델 학습
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

def setup_logging(log_dir, fruit_name=None):
    """
    로깅 설정
    학습 과정을 추적하고 문제 발생 시 디버깅에 도움
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 과실명이 있으면 로그 파일명에 추가
    fruit_suffix = f"_{fruit_name}" if fruit_name else ""
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}{fruit_suffix}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # 콘솔에도 출력
        ]
    )
    
    logger = logging.getLogger(__name__)
    if fruit_name:
        logger.info(f"🍎 과실 종류: {fruit_name}")
    
    return logger

def verify_gpu_availability():
    """
    GPU 사용 가능 여부 확인
    GPU가 있으면 학습 속도가 현저히 빨라집니다
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        
        print(f"✅ GPU 사용 가능: {gpu_count}개 GPU 감지")
        print(f"현재 GPU: {gpu_name}")
        print(f"CUDA 버전: {torch.version.cuda}")
        
        # GPU 메모리 확인
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        print(f"GPU 메모리: {total_memory / 1024**3:.1f}GB")
        
        return True
    else:
        print("⚠️  GPU를 사용할 수 없습니다. CPU로 학습하면 시간이 오래 걸립니다.")
        return False

def load_and_verify_config(data_yaml_path):
    """
    데이터 설정 파일 로드 및 검증
    잘못된 경로나 설정으로 인한 오류를 미리 방지
    """
    data_yaml_path = Path(data_yaml_path)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml 파일을 찾을 수 없습니다: {data_yaml_path}")
    
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 필수 키 확인
    required_keys = ['path', 'train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"data.yaml에 필수 키가 없습니다: {key}")
    
    # 경로 존재 확인
    base_path = Path(config['path'])
    for split in ['train', 'val']:
        if split in config:
            split_path = base_path / config[split]
            if not split_path.exists():
                raise FileNotFoundError(f"경로가 존재하지 않습니다: {split_path}")
    
    # 클래스 수 일치 확인
    if len(config['names']) != config['nc']:
        raise ValueError(f"클래스 수 불일치: nc={config['nc']}, names 길이={len(config['names'])}")
    
    print("✅ 데이터 설정 검증 완료")
    print(f"   - 클래스 수: {config['nc']}")
    print(f"   - 클래스: {config['names']}")
    print(f"   - 데이터 경로: {config['path']}")
    
    return config

# https://docs.ultralytics.com/ko/usage/cfg/#train-settings
# https://docs.ultralytics.com/guides/yolo-data-augmentation/#mosaic-mosaic

def create_training_config(
    model_size='s',  # nano, small, medium, large, xlarge
    epochs=10,
    batch_size=32,
    imgsz=1080,
    optimizer='SGD',
    flipud=0.7, 
    mixup=0.7,
    #single_cls=True,
    lr0=1E-2,
    lrf=0.01,  # final learning rate (lr0 * lrf)
    momentum=0.937,
    weight_decay=0.0005,# 기본값
    warmup_epochs=3, # 기본값
    warmup_momentum=0.8, # 기본값
    warmup_bias_lr=0.1, # 기본값
    box_gain=7.5,  # box loss gain # 기본값
    cls_gain=0.5,   # class loss gain # 기본값
    dfl_gain=1.5,   # distribution focal loss gain # 기본값
    save_period=-1,  # save model every x epochs (-1 to disable)
    workers=8,      # number of worker threads for data loading
    **kwargs
):
    """
    학습 설정 구성
    각 하이퍼파라미터의 의미와 최적값을 이해하는 것이 중요합니다
    
    Args:
        model_size: 모델 크기 (n, s, m, l, x)
                   n(nano): 가장 빠르고 가벼움, 정확도는 낮음
                   s(small): 균형잡힌 성능
                   m(medium): 더 높은 정확도, 적당한 속도
                   l(large): 높은 정확도, 느린 속도  
                   x(xlarge): 최고 정확도, 가장 느림
        epochs: 학습 에포크 수 (전체 데이터를 몇 번 반복할지)
        batch_size: 배치 크기 (한 번에 처리할 이미지 수)
                   GPU 메모리에 따라 조정 필요
        imgsz: 입력 이미지 크기 (정사각형으로 리사이즈)
        optimizer: 최적화 알고리즘
        lr0: 초기 학습률 (너무 크면 불안정, 너무 작으면 느림)
        lrf: 최종 학습률 비율
    """
    
    config = {
        # 모델 설정
        'model': f'yolov8{model_size}.pt',  # 사전 훈련된 모델 사용
        
        # 학습 기본 설정
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'device': [0, 1, 2, 3],  # 자동으로 GPU/CPU 선택
        
        # 최적화 설정
        'optimizer': optimizer,
        'lr0': lr0,
        'lrf': lrf,
        'momentum': momentum,
        'weight_decay': weight_decay,
        
        # 워밍업 설정 (학습 초기에 안정적으로 시작하기 위함)
        'warmup_epochs': warmup_epochs,
        'warmup_momentum': warmup_momentum,  
        'warmup_bias_lr': warmup_bias_lr,
        
        # 손실 함수 가중치
        'box': box_gain,      # 바운딩 박스 위치 손실
        'cls': cls_gain,      # 분류 손실  
        'dfl': dfl_gain,      # Distribution Focal Loss

        # 데이터 증강 -> 추가
        'flipud' : flipud, 
        'mixup' : mixup,
        #'single_cls' : True,
        
        # 데이터 로딩
        'workers': workers,
        
        # 저장 설정
        'save': True,         # 최고 성능 모델 저장
        'save_period': save_period,
        
        # 검증 설정
        'val': True,          # 에포크마다 검증 수행
        
        # 기타 설정
        'verbose': True,      # 상세 로그 출력
        'seed': 42,           # 재현 가능한 결과를 위한 시드
        'exist_ok': True,     # 기존 결과 폴더 덮어쓰기 허용
    }
    
    # 추가 설정 병합
    config.update(kwargs)
    
    return config

def train_yolov8(data_yaml_path, output_dir, config, fruit_name=None):
    """
    YOLOv8 모델 학습 실행
    
    Args:
        data_yaml_path: 데이터셋 설정 파일 경로
        output_dir: 학습 결과 저장 경로  
        config: 학습 설정 딕셔너리
        fruit_name: 과실명 (선택사항)
    """
    
    logger = logging.getLogger(__name__)
    
    # 출력 디렉토리 설정
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 학습 실행 디렉토리 생성 (타임스탬프 + 과실명 포함)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fruit_suffix = f"_{fruit_name}" if fruit_name else ""
    run_name = f"train_{timestamp}{fruit_suffix}"
    run_dir = output_dir / run_name
    
    logger.info(f"학습 시작: {run_name}")
    logger.info(f"출력 경로: {run_dir}")
    
    try:
        # YOLOv8 모델 로드
        model_name = config['model']
        logger.info(f"모델 로드: {model_name}")
        
        model = YOLO(model_name)  # 사전 훈련된 모델 로드
        
        # 모델 정보 출력
        logger.info(f"모델 구조:")
        logger.info(f"  - Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        logger.info(f"  - Trainable params: {sum(p.numel() for p in model.model.parameters() if p.requires_grad):,}")
        
        # 설정을 별도 파일로 저장 (나중에 재현하기 위해) - 과실명 포함
        config_filename = f"training_config{fruit_suffix}.yaml"
        config_save_path = run_dir / config_filename
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # 과실명도 설정에 포함
        config_to_save = config.copy()
        if fruit_name:
            config_to_save['fruit_name'] = fruit_name
        
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_save, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"학습 설정 저장: {config_save_path}")
        
        # 학습 실행
        logger.info("학습 실행 중...")
        
        # project와 name 설정을 통해 결과 저장 위치 지정
        results = model.train(
            data=data_yaml_path,
            project=str(output_dir),
            name=run_name,
            **{k: v for k, v in config.items() if k != 'model'}
        )
        
        logger.info("✅ 학습 완료!")
        
        # 학습 결과 요약
        final_model_path = run_dir / 'weights' / 'best.pt'
        if final_model_path.exists():
            logger.info(f"최고 성능 모델: {final_model_path}")
            
            # 최고 성능 모델을 과실명 포함한 이름으로 복사
            if fruit_name:
                best_model_copy = run_dir / 'weights' / f'best_{fruit_name}.pt'
                shutil.copy2(final_model_path, best_model_copy)
                logger.info(f"과실명 포함 모델: {best_model_copy}")
        
        # 학습 곡선 이미지가 생성되었는지 확인
        results_png = run_dir / 'results.png'
        if results_png.exists():
            logger.info(f"학습 결과 그래프: {results_png}")
            
            # 결과 그래프도 과실명 포함한 이름으로 복사
            if fruit_name:
                results_copy = run_dir / f'results_{fruit_name}.png'
                shutil.copy2(results_png, results_copy)
                logger.info(f"과실명 포함 결과 그래프: {results_copy}")
        
        # confusion matrix도 복사
        confusion_matrix_png = run_dir / 'confusion_matrix.png'
        if confusion_matrix_png.exists() and fruit_name:
            confusion_copy = run_dir / f'confusion_matrix_{fruit_name}.png'
            shutil.copy2(confusion_matrix_png, confusion_copy)
            logger.info(f"과실명 포함 Confusion Matrix: {confusion_copy}")
        
        return run_dir
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 과실 성숙도 검출 모델 학습')
    
    # 필수 인자
    parser.add_argument('--data', required=True, help='data.yaml 파일 경로')
    parser.add_argument('--output_dir', required=True, help='학습 결과 저장 경로')
    
    # 과실명 인자 추가 (선택사항)
    parser.add_argument('--fruit', type=str, help='과실명 (선택사항) - 결과 파일명에 추가됩니다')
    
    # 모델 설정
    parser.add_argument('--model_size', default='s', choices=['n', 's', 'm', 'l', 'x'],
                       help='모델 크기 (n: nano, s: small, m: medium, l: large, x: xlarge)')
    
    # 학습 하이퍼파라미터
    parser.add_argument('--epochs', type=int, default=10, help='학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--imgsz', type=int, default=1080, help='입력 이미지 크기')
    parser.add_argument('--lr0', type=float, default=1E-2, help='초기 학습률')
    
    # GPU 설정
    parser.add_argument('--device', default='auto', help='디바이스 (auto, 0, 1, cpu)')
    
    # 기타 설정
    parser.add_argument('--workers', type=int, default=8, help='데이터 로딩 워커 수')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    
    args = parser.parse_args()
    
    if args.device not in ("auto", "cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # 로깅 설정 (과실명 포함)
    logger = setup_logging(args.output_dir, args.fruit)
    
    try:
        # GPU 확인
        gpu_available = verify_gpu_availability()
        
        # 데이터 설정 검증
        data_config = load_and_verify_config(args.data)
        
        # 학습 설정 구성
        training_config = create_training_config(
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            lr0=args.lr0,
            workers=args.workers,
            device=args.device,
            seed=args.seed
        )
        
        logger.info("=== 학습 설정 요약 ===")
        for key, value in training_config.items():
            logger.info(f"{key}: {value}")
        
        # 학습 실행 (과실명 포함)
        run_dir = train_yolov8(args.data, args.output_dir, training_config, args.fruit)
        
        success_msg = f"🎉 학습 완료! 결과는 다음 경로에 저장되었습니다: {run_dir}"
        if args.fruit:
            success_msg += f"\n🍎 과실: {args.fruit}"
        
        logger.info(success_msg)
        
    except Exception as e:
        logger.error(f"💥 학습 실패: {e}")
        raise

if __name__ == "__main__":
    main()


