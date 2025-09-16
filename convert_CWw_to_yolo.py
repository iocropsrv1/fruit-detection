#!/usr/bin/env python3
"""
두 개의 데이터셋을 합쳐서 새로운 YOLO 데이터셋 생성
- 데이터셋 1: 이미 분할된 YOLO 형식 (/data/ioCrops/<과실명>/dataset/train_v1.1_fruit/)
- 데이터셋 2: COCO JSON 형식 (/home/cat123/crowdworks_20250610/detection/)
- 결과: 과실별로 합쳐진 새로운 데이터셋 (8:1:1 분할)
"""

import json
import shutil
from pathlib import Path
import argparse
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ===== 클래스 정규화/통합 규칙 =====
TARGET_NAMES_DEFAULT = ['ripened', 'ripening', 'unripened']
TARGET_NAME_MAP = {
    'ripened': 'ripened',
    'ripening': 'ripening',
    'unripened': 'unripened',
    'unripe': 'unripened',
    'notripened': 'unripened',
    'un_ripened': 'unripened',
    'un-ripened': 'unripened',
}

def normalize_class_name(name: str) -> str:
    return TARGET_NAME_MAP.get(name.strip().lower().replace(' ', '').replace('\t', ''), name.strip().lower())

def convert_bbox_coco_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return [cx, cy, w / img_w, h / img_h]

def ensure_dirs_for_fruit(base: Path):
    # staging
    (base / 'images' / 'all').mkdir(parents=True, exist_ok=True)
    (base / 'labels' / 'all').mkdir(parents=True, exist_ok=True)
    # final splits
    for split in ['train', 'val', 'test']:
        (base / 'images' / split).mkdir(parents=True, exist_ok=True)
        (base / 'labels' / split).mkdir(parents=True, exist_ok=True)

def create_data_yaml(dataset_dir: Path,
                     yaml_path_override: str,
                     yaml_train_rel: str,
                     yaml_val_rel: str,
                     yaml_test_rel: str,
                     target_names):
    content = (
        f"path: {yaml_path_override}\n"
        f"train: {yaml_train_rel}\n"
        f"val: {yaml_val_rel}\n"
        f"test: {yaml_test_rel}\n\n"
        f"nc: {len(target_names)}\n"
        f"names: {target_names}\n"
    )
    with open(dataset_dir / 'data.yaml', 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[data.yaml] {dataset_dir/'data.yaml'}")

# ===== 데이터셋 2 수집 (COCO JSON) =====
def collect_dataset2_items(input_root: Path):
    """
    데이터셋 2에서 아이템 수집
    반환: items_by_fruit[fruit] = list of dict(
        image_path, json_path, fruit, region, date, section, file_name
    )
    """
    items_by_fruit = defaultdict(list)
    images_root = input_root / 'images'
    labels_root = input_root / 'labels'
    
    if not images_root.exists() or not labels_root.exists():
        print(f"[Warn] Dataset2 not found: {input_root}")
        return items_by_fruit

    for fruit_dir in images_root.iterdir():
        if not fruit_dir.is_dir():
            continue
        fruit = fruit_dir.name
        for region_dir in fruit_dir.iterdir():
            if not region_dir.is_dir():
                continue
            region = region_dir.name
            for date_dir in region_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                date = date_dir.name
                for section_dir in date_dir.iterdir():
                    if not section_dir.is_dir():
                        continue
                    section = section_dir.name
                    json_path = labels_root / fruit / region / date / f"{section}.json"
                    if not json_path.exists():
                        print(f"[Warn] missing label: {json_path}")
                        continue
                    for img_path in section_dir.glob("*.png"):
                        items_by_fruit[fruit].append({
                            'image_path': img_path,
                            'json_path': json_path,
                            'fruit': fruit,
                            'region': region,
                            'date': date,
                            'section': section,
                            'file_name': img_path.name
                        })
    return items_by_fruit

# ===== 데이터셋 1 수집 (이미 분할된 YOLO) =====
def collect_dataset1_items(dataset1_root: Path):
    """
    데이터셋 1에서 아이템 수집 (이미 train/valid/test로 분할됨)
    반환: items_by_fruit[fruit][split] = list of dict(image_path, label_path, file_name)
    """
    items_by_fruit = defaultdict(lambda: defaultdict(list))
    
    if not dataset1_root.exists():
        print(f"[Warn] Dataset1 root not found: {dataset1_root}")
        return items_by_fruit
    
    # 과실별로 순회
    for fruit in ['pepper', 'tomato', 'berry']:
        fruit_path = dataset1_root / fruit / 'dataset' / 'train_v1.1_fruit'
        if not fruit_path.exists():
            print(f"[Warn] Dataset1 fruit path not found: {fruit_path}")
            continue
            
        # train, valid, test별로 순회
        for split in ['train', 'valid', 'test']:
            images_dir = fruit_path / 'images' / split
            labels_dir = fruit_path / 'labels' / split
            
            if not images_dir.exists() or not labels_dir.exists():
                print(f"[Warn] Dataset1 split not found: {images_dir} or {labels_dir}")
                continue
                
            for img_path in images_dir.glob("*.png"):
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    items_by_fruit[fruit][split].append({
                        'image_path': img_path,
                        'label_path': label_path,
                        'file_name': img_path.name
                    })
    
    return items_by_fruit

# ===== COCO -> YOLO (데이터셋 2) =====
def convert_dataset2_to_staging_all(items, fruit_out_dir: Path, target_names, strict_drop_unknown=True):
    """
    데이터셋 2의 각 이미지에 대해 YOLO 라벨 생성하여
    <fruit>/images/all, <fruit>/labels/all 에 저장
    파일명: ds2_fruit_region_date_section_original.png(.txt)
    """
    ensure_dirs_for_fruit(fruit_out_dir)
    class_map = {name: idx for idx, name in enumerate(target_names)}
    coco_cache = {}

    kept = 0
    dropped = 0

    for info in items:
        img_path = info['image_path']
        json_path = info['json_path']

        if json_path not in coco_cache:
            with open(json_path, 'r', encoding='utf-8') as f:
                coco_cache[json_path] = json.load(f)
        coco = coco_cache[json_path]

        # 인덱스
        images_info = {img['id']: img for img in coco.get('images', [])}
        anns_by_image = defaultdict(list)
        for ann in coco.get('annotations', []):
            anns_by_image[ann['image_id']].append(ann)

        cat_name_by_id = {}
        for cat in coco.get('categories', []):
            cat_name_by_id[cat['id']] = normalize_class_name(cat.get('name', ''))

        # 매칭
        matched_img = None
        matched_anns = []
        for img_id, meta in images_info.items():
            if meta.get('file_name') == info['file_name']:
                matched_img = meta
                matched_anns = anns_by_image[img_id]
                break
        if not matched_img:
            print(f"[Warn] no ann for {info['file_name']}")
            continue

        # 목적지 파일명(충돌 방지 - 데이터셋 2 접두사 추가)
        base_name = f"ds2_{info['fruit']}_{info['region']}_{info['date']}_{info['section']}_{info['file_name']}"
        dst_img = fruit_out_dir / 'images' / 'all' / base_name
        shutil.copy2(img_path, dst_img)

        dst_lbl = fruit_out_dir / 'labels' / 'all' / (dst_img.stem + '.txt')

        # 라벨 작성
        count_lines = 0
        with open(dst_lbl, 'w', encoding='utf-8') as lf:
            for ann in matched_anns:
                if 'category_id' not in ann or 'bbox' not in ann:
                    continue
                cname = cat_name_by_id.get(ann['category_id'])
                if cname not in class_map:
                    if strict_drop_unknown:
                        continue
                    else:
                        continue
                cls_id = class_map[cname]
                w, h = matched_img['width'], matched_img['height']
                cx, cy, bw, bh = convert_bbox_coco_to_yolo(ann['bbox'], w, h)
                lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                count_lines += 1

        if count_lines > 0:
            kept += 1
        else:
            dropped += 1

    print(f"[Dataset2 Staging] {fruit_out_dir.name}: kept(>=1 box)={kept}, empty_after_filter={dropped}")

# ===== 데이터셋 1 복사 (staging) =====
def copy_dataset1_to_staging_all(items_by_split, fruit_out_dir: Path):
    """
    데이터셋 1의 파일들을 staging(all) 디렉터리로 복사
    파일명: ds1_<split>_<original_name>
    """
    ensure_dirs_for_fruit(fruit_out_dir)
    
    total_copied = 0
    for split, items in items_by_split.items():
        for item in items:
            # 충돌 방지를 위해 데이터셋 1 접두사 및 split 정보 추가
            base_name = f"ds1_{split}_{item['file_name']}"
            
            # 이미지 복사
            dst_img = fruit_out_dir / 'images' / 'all' / base_name
            shutil.copy2(item['image_path'], dst_img)
            
            # 라벨 복사
            dst_lbl = fruit_out_dir / 'labels' / 'all' / (dst_img.stem + '.txt')
            shutil.copy2(item['label_path'], dst_lbl)
            
            total_copied += 1
    
    print(f"[Dataset1 Staging] {fruit_out_dir.name}: copied {total_copied} files")

# ===== 랜덤 분할: 0.8 / 0.1 / 0.1 =====
def random_split_80_10_10_and_materialize(fruit_out_dir: Path, seed: int):
    """
    staging(all)에 있는 파일 목록을 기준으로 랜덤 분할 후
    train/val/test 디렉터리로 이미지/라벨을 복사.
    """
    rng = random.Random(seed)

    all_imgs = sorted((fruit_out_dir / 'images' / 'all').glob('*.png'))
    all_basenames = [p.stem for p in all_imgs]

    if len(all_basenames) == 0:
        print(f"[Split] {fruit_out_dir.name}: nothing in staging.")
        return

    # 먼저 test 10% 분할
    names_rest, names_test = train_test_split(all_basenames, test_size=0.1, random_state=seed)
    # 나머지에서 val 10% (전체 기준 → 나머지 대비 비율 = 0.1 / 0.9)
    names_train, names_val = train_test_split(names_rest, test_size=(0.1/0.9), random_state=seed)

    print(f"[Split] {fruit_out_dir.name}: train={len(names_train)}, val={len(names_val)}, test={len(names_test)}")

    def copy_for_split(split_name, names):
        for stem in names:
            src_img = fruit_out_dir / 'images' / 'all' / f"{stem}.png"
            src_lbl = fruit_out_dir / 'labels' / 'all' / f"{stem}.txt"
            dst_img = fruit_out_dir / 'images' / split_name / f"{stem}.png"
            dst_lbl = fruit_out_dir / 'labels' / split_name / f"{stem}.txt"
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)

    copy_for_split('train', names_train)
    copy_for_split('val',   names_val)
    copy_for_split('test',  names_test)

# ===== 메인 =====
def main():
    ap = argparse.ArgumentParser(description="Merge two datasets and create new YOLO dataset")
    ap.add_argument('--dataset1_root', default='/data/ioCrops', 
                    help='Dataset1 root (contains <fruit>/dataset/train_v1.1_fruit/)')
    ap.add_argument('--dataset2_root', default='/home/cat123/crowdworks_20250610/detection', 
                    help='Dataset2 root (COCO format, contains images/, labels/)')
    ap.add_argument('--output_dir', default='/home/cat123/yolov8-fruit_detection/yolo_dataset_merged_new',
                    help='Output root (per fruit subdir will be created)')
    ap.add_argument('--seed', type=int, default=42)

    # data.yaml 스키마
    ap.add_argument('--yaml_path_override', default='/home/cat123/yolov8-fruit_detection/yolo_dataset_merged_new')
    ap.add_argument('--yaml_train_rel', default='images/train/')
    ap.add_argument('--yaml_val_rel', default='images/val/')
    ap.add_argument('--yaml_test_rel', default='images/test/')
    ap.add_argument('--yaml_names', default='ripened,ripening,unripened')

    args = ap.parse_args()
    random.seed(args.seed)

    dataset1_root = Path(args.dataset1_root)
    dataset2_root = Path(args.dataset2_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    target_names = [s.strip() for s in args.yaml_names.split(',') if s.strip()]
    if not target_names:
        target_names = TARGET_NAMES_DEFAULT

    print("=== 데이터셋 수집 ===")
    
    # 1) 데이터셋 2 수집 (COCO JSON)
    print("데이터셋 2 (COCO) 수집...")
    dataset2_items = collect_dataset2_items(dataset2_root)
    
    # 2) 데이터셋 1 수집 (이미 분할된 YOLO)
    print("데이터셋 1 (YOLO) 수집...")
    dataset1_items = collect_dataset1_items(dataset1_root)

    # 3) 모든 과실 목록 수집
    all_fruits = set(dataset2_items.keys()) | set(dataset1_items.keys())
    print(f"처리할 과실: {sorted(all_fruits)}")

    # 4) 과실별 처리
    for fruit in sorted(all_fruits):
        print(f"\n=== 과실: {fruit} ===")
        fruit_out = output_root / fruit
        
        # 데이터셋 2 변환 → staging(all)
        if fruit in dataset2_items:
            print(f"데이터셋 2 변환 중... ({len(dataset2_items[fruit])} items)")
            convert_dataset2_to_staging_all(dataset2_items[fruit], fruit_out, target_names)
        
        # 데이터셋 1 복사 → staging(all)
        if fruit in dataset1_items:
            total_ds1 = sum(len(items) for items in dataset1_items[fruit].values())
            print(f"데이터셋 1 복사 중... ({total_ds1} items)")
            copy_dataset1_to_staging_all(dataset1_items[fruit], fruit_out)
        
        # 랜덤 분할(8:1:1) → train/val/test 디렉터리 생성/복사
        print(f"랜덤 분할 (8:1:1) 중...")
        random_split_80_10_10_and_materialize(fruit_out, seed=args.seed)

        # data.yaml 생성
        create_data_yaml(
            dataset_dir=fruit_out,
            yaml_path_override=f"{args.yaml_path_override}/{fruit}",
            yaml_train_rel=args.yaml_train_rel,
            yaml_val_rel=args.yaml_val_rel,
            yaml_test_rel=args.yaml_test_rel,
            target_names=target_names
        )

    print("\n=== 완료 ===")
    print(f"통합 데이터셋이 생성되었습니다: {output_root}")
    
    # 최종 통계
    print("\n=== 최종 통계 ===")
    for fruit in sorted(all_fruits):
        fruit_out = output_root / fruit
        if fruit_out.exists():
            train_count = len(list((fruit_out / 'images' / 'train').glob('*.png')))
            val_count = len(list((fruit_out / 'images' / 'val').glob('*.png')))
            test_count = len(list((fruit_out / 'images' / 'test').glob('*.png')))
            total_count = train_count + val_count + test_count
            print(f"{fruit}: train={train_count}, val={val_count}, test={test_count}, total={total_count}")

if __name__ == "__main__":
    main()