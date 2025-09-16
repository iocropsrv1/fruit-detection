#!/usr/bin/env python3
"""
COCO 형식을 YOLO 형식으로 변환하는 스크립트
각 과실별로 변환 후 전체 데이터셋을 train/val/test로 분할
"""

import json
import os
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import argparse

def convert_bbox_coco_to_yolo(bbox, img_width, img_height):
    """
    COCO 바운딩 박스 (x, y, width, height) 절대 좌표를
    YOLO 형식 (center_x, center_y, width, height) 정규화 좌표로 변환
    
    Args:
        bbox: [x, y, width, height] COCO 형식
        img_width, img_height: 이미지 크기
    
    Returns:
        [center_x, center_y, width, height] YOLO 형식 (0~1 정규화)
    """
    x, y, w, h = bbox
    
    # 중심점 계산 (COCO의 x,y는 좌상단 기준)
    center_x = x + w / 2
    center_y = y + h / 2
    
    # 정규화 (이미지 크기로 나누기)
    center_x_norm = center_x / img_width
    center_y_norm = center_y / img_height
    width_norm = w / img_width
    height_norm = h / img_height
    
    return [center_x_norm, center_y_norm, width_norm, height_norm]

def coco_to_yolo_single(coco_json_path, images_dir, output_dir):
    """
    단일 COCO JSON 파일을 YOLO 포맷으로 변환
    
    Args:
        coco_json_path: COCO JSON 파일 경로
        images_dir: 이미지 디렉토리 경로
        output_dir: 출력 디렉토리 경로
    
    Returns:
        category_mapping: 클래스 ID 매핑 정보
    """
    
    # 출력 디렉토리 생성
    labels_dir = Path(output_dir) / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO 데이터 로드
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 카테고리 정보 추출 및 ID 재매핑
    categories = coco_data['categories']
    # YOLO는 0부터 시작하는 클래스 ID를 사용
    category_mapping = {}
    class_names = []
    
    for idx, category in enumerate(categories):
        category_mapping[category['id']] = idx  # COCO ID -> YOLO ID
        class_names.append(category['name'])
    
    print(f"클래스 매핑: {category_mapping}")
    print(f"클래스 이름: {class_names}")
    
    # 이미지 정보를 딕셔너리로 구성 (빠른 검색을 위해)
    images_info = {img['id']: img for img in coco_data['images']}
    
    # 어노테이션을 이미지별로 그룹화
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # 각 이미지에 대해 YOLO 라벨 파일 생성
    converted_count = 0
    
    for image_id, annotations in image_annotations.items():
        if image_id not in images_info:
            continue
            
        image_info = images_info[image_id]
        image_filename = image_info['file_name']
        img_width = image_info['width']
        img_height = image_info['height']
        
        # 이미지 파일이 실제로 존재하는지 확인
        image_path = Path(images_dir) / image_filename
        if not image_path.exists():
            print(f"경고: 이미지 파일이 존재하지 않습니다: {image_path}")
            continue
        
        # YOLO 라벨 파일 생성
        label_filename = Path(image_filename).stem + '.txt'
        label_path = labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # 클래스 ID 변환
                coco_class_id = ann['category_id']
                if coco_class_id not in category_mapping:
                    print(f"경고: 알 수 없는 클래스 ID {coco_class_id}")
                    continue
                    
                yolo_class_id = category_mapping[coco_class_id]
                
                # 바운딩 박스 변환
                bbox = ann['bbox']
                yolo_bbox = convert_bbox_coco_to_yolo(bbox, img_width, img_height)
                
                # YOLO 형식으로 저장: class_id center_x center_y width height
                f.write(f"{yolo_class_id} {' '.join(f'{x:.6f}' for x in yolo_bbox)}\n")
        
        converted_count += 1
    
    print(f"변환 완료: {converted_count}개 이미지")
    
    return class_names, category_mapping

def create_dataset_splits(merged_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    변환된 데이터를 train/val/test로 분할하고 YOLO 형식으로 구성
    
    Args:
        merged_dir: 변환된 데이터가 있는 디렉토리
        output_dir: 최종 YOLO 데이터셋이 저장될 디렉토리
        train_ratio, val_ratio, test_ratio: 분할 비율
    """
    
    # 최종 YOLO 데이터셋 구조 생성
    yolo_dataset_dir = Path(output_dir)
    yolo_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # train/val/test 디렉토리 생성
    for split in ['train', 'val', 'test']:
        (yolo_dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (yolo_dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    all_class_names = set()
    
    # 각 과실별로 처리
    fruit_types = ['berry', 'korean_melon', 'pepper', 'tomato']
    
    for fruit in fruit_types:
        fruit_dir = Path(merged_dir) / fruit
        
        if not fruit_dir.exists():
            print(f"경고: {fruit} 디렉토리가 존재하지 않습니다.")
            continue
            
        print(f"\n=== {fruit} YOLO 변환 시작 ===")
        
        # 임시 변환 디렉토리
        temp_yolo_dir = fruit_dir / 'temp_yolo'
        temp_yolo_dir.mkdir(exist_ok=True)
        
        # COCO -> YOLO 변환
        coco_json_path = fruit_dir / 'annotations' / f'{fruit}_merged.json'
        images_dir = fruit_dir / 'images'
        
        if not coco_json_path.exists():
            print(f"경고: COCO JSON 파일이 존재하지 않습니다: {coco_json_path}")
            continue
            
        class_names, category_mapping = coco_to_yolo_single(
            coco_json_path, images_dir, temp_yolo_dir
        )
        
        all_class_names.update(class_names)
        
        # 이미지와 라벨 파일 목록 가져오기
        label_files = list((temp_yolo_dir / 'labels').glob('*.txt'))
        
        # 빈 라벨 파일 제거 (객체가 없는 이미지)
        valid_label_files = []
        for label_file in label_files:
            if label_file.stat().st_size > 0:  # 파일이 비어있지 않은 경우
                valid_label_files.append(label_file)
            else:
                print(f"빈 라벨 파일 제거: {label_file}")
        
        if not valid_label_files:
            print(f"경고: {fruit}에서 유효한 라벨 파일을 찾을 수 없습니다.")
            continue
        
        # train/val/test 분할
        # 먼저 train과 temp(val+test)로 분할
        train_files, temp_files = train_test_split(
            valid_label_files, 
            test_size=(val_ratio + test_ratio),
            random_state=42
        )
        
        # temp를 val과 test로 분할
        val_files, test_files = train_test_split(
            temp_files,
            test_size=test_ratio/(val_ratio + test_ratio),
            random_state=42
        )
        
        print(f"데이터 분할 결과:")
        print(f"  Train: {len(train_files)}개")
        print(f"  Validation: {len(val_files)}개") 
        print(f"  Test: {len(test_files)}개")
        
        # 각 분할에 대해 파일 복사
        splits = {
            'train': train_files,
            'val': val_files, 
            'test': test_files
        }
        
        for split_name, files in splits.items():
            for label_file in files:
                # 라벨 파일 복사
                target_label_path = yolo_dataset_dir / split_name / 'labels' / label_file.name
                shutil.copy2(label_file, target_label_path)
                
                # 해당하는 이미지 파일 복사
                image_name = label_file.stem
                # 가능한 확장자들을 확인
                for ext in ['.png', '.jpg', '.jpeg']:
                    source_image_path = images_dir / f"{image_name}{ext}"
                    if source_image_path.exists():
                        target_image_path = yolo_dataset_dir / split_name / 'images' / f"{image_name}{ext}"
                        shutil.copy2(source_image_path, target_image_path)
                        break
        
        # 임시 디렉토리 정리
        shutil.rmtree(temp_yolo_dir)
    
    # data.yaml 파일 생성
    create_data_yaml(yolo_dataset_dir, sorted(list(all_class_names)))
    
    print(f"\n=== YOLO 데이터셋 생성 완료 ===")
    print(f"출력 경로: {yolo_dataset_dir}")
    
    # 최종 통계 출력
    for split in ['train', 'val', 'test']:
        image_count = len(list((yolo_dataset_dir / split / 'images').glob('*')))
        label_count = len(list((yolo_dataset_dir / split / 'labels').glob('*.txt')))
        print(f"{split.capitalize()}: {image_count}개 이미지, {label_count}개 라벨")

def create_data_yaml(dataset_dir, class_names):
    """
    YOLO 학습용 data.yaml 파일 생성
    
    Args:
        dataset_dir: 데이터셋 디렉토리
        class_names: 클래스 이름 리스트
    """
    
    data_yaml_content = f"""# YOLO 데이터셋 설정
path: {dataset_dir.absolute()}  # 데이터셋 루트 경로
train: train/images  # 훈련 이미지 경로 (path 기준 상대 경로)
val: val/images      # 검증 이미지 경로
test: test/images    # 테스트 이미지 경로

# 클래스 정보
nc: {len(class_names)}  # 클래스 개수
names: {class_names}    # 클래스 이름
"""
    
    yaml_path = dataset_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(data_yaml_content)
    
    print(f"data.yaml 생성 완료: {yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COCO를 YOLO 포맷으로 변환')
    parser.add_argument('--merged_dir', required=True, help='통합된 데이터 디렉토리')
    parser.add_argument('--output_dir', required=True, help='YOLO 데이터셋 출력 디렉토리')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='훈련 데이터 비율')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='검증 데이터 비율')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='테스트 데이터 비율')
    
    args = parser.parse_args()
    
    # 비율 검증
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"데이터 분할 비율의 합이 1이 아닙니다: {total_ratio}")
    
    print("COCO to YOLO 변환을 시작합니다...")
    create_dataset_splits(args.merged_dir, args.output_dir, 
                         args.train_ratio, args.val_ratio, args.test_ratio)
    print("변환 완료!")