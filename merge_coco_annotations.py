#!/usr/bin/env python3
"""
COCO 어노테이션 파일 통합 스크립트
여러 JSON 파일을 하나로 합치고 이미지도 해당 폴더로 이동
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
import argparse

def merge_coco_annotations(source_dir, output_dir):
    """
    과실별로 흩어진 COCO JSON 파일들을 하나로 통합하는 함수
    
    Args:
        source_dir: crowdworks_20250610/detection 경로
        output_dir: 통합된 데이터가 저장될 경로
    """
    
    # 과실 종류별로 데이터 수집
    fruit_types = ['berry', 'korean_melon', 'pepper', 'tomato']
    
    for fruit in fruit_types:
        print(f"\n=== {fruit} 데이터 처리 시작 ===")
        
        # 출력 디렉토리 생성
        fruit_output_dir = Path(output_dir) / fruit
        images_output_dir = fruit_output_dir / 'images'
        annotations_output_dir = fruit_output_dir / 'annotations'
        
        images_output_dir.mkdir(parents=True, exist_ok=True)
        annotations_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 통합할 COCO 데이터 구조 초기화
        merged_coco = {
            "info": {
                "description": f"{fruit} dataset for YOLOv8 training",
                "version": "1.0",
                "contributor": "Auto-generated"
            },
            "licenses": [],
            "categories": [],
            "images": [],
            "annotations": []
        }
        
        # 카테고리 정보 (첫 번째 JSON에서 가져오기)
        categories_set = False
        
        # ID 매핑을 위한 카운터
        image_id_counter = 1
        annotation_id_counter = 1
        
        # 이미지 파일명 중복 방지를 위한 세트
        processed_images = set()
        
        # 각 과실의 모든 JSON 파일 찾기
        labels_dir = Path(source_dir) / 'labels' / fruit
        images_dir = Path(source_dir) / 'images' / fruit
        
        if not labels_dir.exists():
            print(f"경고: {labels_dir} 경로가 존재하지 않습니다.")
            continue
            
        # JSON 파일들을 재귀적으로 찾기
        json_files = list(labels_dir.rglob('*.json'))
        print(f"{fruit}에서 {len(json_files)}개의 JSON 파일을 찾았습니다.")
        
        for json_file in json_files:
            print(f"처리 중: {json_file}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                
                # 카테고리 정보 설정 (한 번만)
                if not categories_set:
                    merged_coco["categories"] = coco_data.get("categories", [])
                    merged_coco["licenses"] = coco_data.get("licenses", [])
                    categories_set = True
                
                # 해당 JSON과 연결된 이미지 폴더 찾기
                json_relative_path = json_file.relative_to(labels_dir)
                json_stem = json_file.stem  # 확장자 제거한 파일명
                
                # 이미지 폴더 경로 구성
                corresponding_image_dir = images_dir / json_relative_path.parent / json_stem
                
                if not corresponding_image_dir.exists():
                    print(f"경고: {corresponding_image_dir} 이미지 폴더가 존재하지 않습니다.")
                    continue
                
                # 이미지 파일들 처리
                image_files = list(corresponding_image_dir.glob('*.png'))
                print(f"  - {len(image_files)}개의 이미지 파일 발견")
                
                # 기존 image ID와 새 image ID 매핑
                old_to_new_image_id = {}
                
                for image_info in coco_data.get("images", []):
                    original_image_id = image_info["id"]
                    original_filename = image_info["file_name"]
                    
                    # 실제 이미지 파일 경로
                    original_image_path = corresponding_image_dir / original_filename
                    
                    if not original_image_path.exists():
                        print(f"경고: 이미지 파일 {original_image_path}를 찾을 수 없습니다.")
                        continue
                    
                    # 파일명 중복 방지를 위한 고유 이름 생성
                    base_name = original_filename
                    counter = 1
                    while base_name in processed_images:
                        name, ext = os.path.splitext(original_filename)
                        base_name = f"{name}_{counter}{ext}"
                        counter += 1
                    
                    processed_images.add(base_name)
                    
                    # 이미지 복사
                    target_image_path = images_output_dir / base_name
                    shutil.copy2(original_image_path, target_image_path)
                    
                    # 새로운 이미지 정보 생성
                    new_image_info = image_info.copy()
                    new_image_info["id"] = image_id_counter
                    new_image_info["file_name"] = base_name
                    
                    merged_coco["images"].append(new_image_info)
                    old_to_new_image_id[original_image_id] = image_id_counter
                    
                    image_id_counter += 1
                
                # 어노테이션 정보 처리
                for annotation in coco_data.get("annotations", []):
                    if annotation["image_id"] in old_to_new_image_id:
                        new_annotation = annotation.copy()
                        new_annotation["id"] = annotation_id_counter
                        new_annotation["image_id"] = old_to_new_image_id[annotation["image_id"]]
                        
                        merged_coco["annotations"].append(new_annotation)
                        annotation_id_counter += 1
                
            except Exception as e:
                print(f"오류 발생 {json_file}: {e}")
                continue
        
        # 통합된 COCO JSON 저장
        output_json_path = annotations_output_dir / f"{fruit}_merged.json"
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(merged_coco, f, indent=2, ensure_ascii=False)
        
        print(f"{fruit} 완료:")
        print(f"  - 총 이미지: {len(merged_coco['images'])}개")
        print(f"  - 총 어노테이션: {len(merged_coco['annotations'])}개")
        print(f"  - 출력 경로: {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='COCO 어노테이션 파일 통합')
    parser.add_argument('--source', required=True, help='소스 디렉토리 (crowdworks_20250610/detection)')
    parser.add_argument('--output', required=True, help='출력 디렉토리')
    
    args = parser.parse_args()
    
    print("COCO 어노테이션 통합을 시작합니다...")
    merge_coco_annotations(args.source, args.output)
    print("통합 완료!")