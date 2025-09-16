#!/usr/bin/env python3
"""
Crowdworks -> YOLO 변환 (과실별 분리 저장, Stratified K-Fold 분할)
- 변환: COCO(JSON) → YOLO(txt)  (staging: images/all, labels/all)
- 분할:
    1) Stratified Holdout Test 10%
    2) 남은 90%에 대해 StratifiedKFold (기본 n_splits=9) → --fold 인덱스가 val
- 결과: <output_dir>/<fruit>/
          images/{train,val,test}/
          labels/{train,val,test}/
          data.yaml
"""

import json
import shutil
from pathlib import Path
import argparse
import random
from collections import defaultdict, Counter

from sklearn.model_selection import train_test_split, StratifiedKFold

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

# ===== 데이터 수집 =====
def collect_items(input_root: Path):
    """
    반환: items_by_fruit[fruit] = list of dict(
        image_path, json_path, fruit, region, date, section, file_name
    )
    """
    items_by_fruit = defaultdict(list)
    images_root = input_root / 'images'
    labels_root = input_root / 'labels'
    if not images_root.exists() or not labels_root.exists():
        raise FileNotFoundError(f"images/labels not found under {input_root}")

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

# ===== COCO -> YOLO (staging: all) =====
def convert_items_to_staging_all(items, fruit_out_dir: Path, target_names, strict_drop_unknown=True):
    """
    각 이미지에 대해 YOLO 라벨 생성하여
    <fruit>/images/all, <fruit>/labels/all 에 저장
    파일명: fruit_region_date_section_original.png(.txt)
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

        # 목적지 파일명(충돌 방지)
        base_name = f"{info['fruit']}_{info['region']}_{info['date']}_{info['section']}_{info['file_name']}"
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
            # 빈 라벨도 그대로 둠 (YOLO 허용). 카운트만 기록
            dropped += 1

    print(f"[Staging] {fruit_out_dir.name}: kept(>=1 box)={kept}, empty_after_filter={dropped}")

# ===== 라벨 읽어 이미지별 대표 클래스 산출 =====
def build_stratify_labels_from_staging(fruit_out_dir: Path, num_classes: int):
    """
    labels/all/*.txt 를 읽어 이미지별 '대표 클래스'(가장 많이 등장한 클래스 id)를 반환
    - 라벨이 비면 대표 클래스를 None으로 표기
    반환: stems(list[str]), y(list[int or None])
    """
    label_files = sorted((fruit_out_dir / 'labels' / 'all').glob('*.txt'))
    stems, y = [], []
    for lf in label_files:
        counts = Counter()
        with open(lf, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                if 0 <= cid < num_classes:
                    counts[cid] += 1
        stems.append(lf.stem)
        if len(counts) == 0:
            y.append(None)  # empty label
        else:
            y.append(counts.most_common(1)[0][0])
    return stems, y

def _adjust_n_splits_for_stratify(y, desired_n_splits):
    """
    각 클래스 빈도 확인 후 n_splits를 안전하게 조정 (최소 빈도 이상이어야 함)
    None(빈 라벨)은 별도 처리 → 가장 희소한 실제 클래스에 흡수하여 분할 안정화
    """
    # None 제거하고 실제 클래스 빈도만 본다
    counts = Counter([c for c in y if c is not None])
    if not counts:
        return 2  # 최저 폴드
    min_count = min(counts.values())
    n_splits = min(desired_n_splits, max(2, min_count))
    return n_splits

def _remap_none_labels_to_rare_class(y):
    """None(빈 라벨)을 가장 희소한 실제 클래스로 치환하여 stratify가 가능하도록 함."""
    counts = Counter([c for c in y if c is not None])
    if not counts:
        # 모두 None이면 0으로 통일
        return [0 if c is None else c for c in y]
    rare_cls = min(counts, key=lambda k: counts[k])
    return [rare_cls if c is None else c for c in y]

# ===== materialize (복사) =====
def _copy_by_stems(fruit_out_dir: Path, split_name: str, stems: list[str]):
    for stem in stems:
        src_img = fruit_out_dir / 'images' / 'all' / f"{stem}.png"
        src_lbl = fruit_out_dir / 'labels' / 'all' / f"{stem}.txt"
        dst_img = fruit_out_dir / 'images' / split_name / f"{stem}.png"
        dst_lbl = fruit_out_dir / 'labels' / split_name / f"{stem}.txt"
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)

# ===== Stratified 분할 파이프라인 =====
def stratified_kfold_split_and_materialize(fruit_out_dir: Path, seed: int, n_splits: int, fold_index: int):
    """
    1) 전체 -> test 10% (Stratified)
    2) 남은 -> StratifiedKFold(n_splits) → fold_index를 val, 나머지를 train
    """
    # 1) 라벨 생성
    stems, y = build_stratify_labels_from_staging(fruit_out_dir, num_classes=3)
    if len(stems) == 0:
        print(f"[Split] {fruit_out_dir.name}: nothing in staging.")
        return

    # None(빈 라벨) 치환
    y_fill = _remap_none_labels_to_rare_class(y)

    # 2) test 10% stratified
    stems_rest, stems_test, y_rest, y_test = train_test_split(
        stems, y_fill, test_size=0.10, random_state=seed, stratify=y_fill
    )

    # 3) StratifiedKFold on rest
    n_splits_safe = _adjust_n_splits_for_stratify(y_rest, n_splits)
    if fold_index >= n_splits_safe:
        print(f"[Warn] fold_index {fold_index} >= n_splits {n_splits_safe}. Using fold_index=0.")
        fold_index = 0

    skf = StratifiedKFold(n_splits=n_splits_safe, shuffle=True, random_state=seed)
    # stems_rest 의 인덱스 기반으로 수행
    idx_rest = list(range(len(stems_rest)))
    # stratify 용 라벨
    y_rest_list = list(y_rest)

    train_idx, val_idx = None, None
    for i, (tr, va) in enumerate(skf.split(idx_rest, y_rest_list)):
        if i == fold_index:
            train_idx, val_idx = tr, va
            break

    stems_train = [stems_rest[i] for i in train_idx]
    stems_val   = [stems_rest[i] for i in val_idx]

    print(f"[Split] {fruit_out_dir.name}: "
          f"train={len(stems_train)}, val={len(stems_val)}, test={len(stems_test)} "
          f"(n_splits={n_splits_safe}, fold={fold_index})")

    # 4) 복사
    _copy_by_stems(fruit_out_dir, 'train', stems_train)
    _copy_by_stems(fruit_out_dir, 'val',   stems_val)
    _copy_by_stems(fruit_out_dir, 'test',  stems_test)

# ===== 메인 =====
def main():
    ap = argparse.ArgumentParser(description="COCO->YOLO per-fruit, then Stratified K-Fold split")
    ap.add_argument('--input_dir', required=True, help='Input root (images/, labels/ under it)')
    ap.add_argument('--output_dir', default='/home/cat123/yolov8-fruit_detection/yolo_dataset_new_stratifield',
                    help='Output root (per fruit subdir will be created)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_splits', type=int, default=9, help='Stratified K-Fold splits (default 9 ≈ 10% val)')
    ap.add_argument('--fold', type=int, default=0, help='Which fold to use as validation')

    # data.yaml 스키마(기본: 일반 구성)
    ap.add_argument('--yaml_path_override', default='/data/ioCrops/berry/dataset/fruit/train_v1.0')
    ap.add_argument('--yaml_train_rel', default='images/train/')
    ap.add_argument('--yaml_val_rel', default='images/val/')
    ap.add_argument('--yaml_test_rel', default='images/test/')
    ap.add_argument('--yaml_names', default='ripened,ripening,unripened')

    args = ap.parse_args()
    random.seed(args.seed)

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    target_names = [s.strip() for s in args.yaml_names.split(',') if s.strip()]
    if not target_names:
        target_names = TARGET_NAMES_DEFAULT

    # 1) 아이템 수집
    items_by_fruit = collect_items(input_root)

    # 2) 과실별 변환 → staging(all)
    for fruit, items in sorted(items_by_fruit.items()):
        fruit_out = output_root / fruit
        print(f"\n=== Convert to YOLO (staging) | fruit={fruit}, N={len(items)} ===")
        convert_items_to_staging_all(items, fruit_out, target_names, strict_drop_unknown=True)

    # 3) 과실별 Stratified 분할 (test 10% + KFold val)
    for fruit in sorted(items_by_fruit.keys()):
        fruit_out = output_root / fruit
        print(f"\n=== Stratified split | fruit={fruit} ===")
        ensure_dirs_for_fruit(fruit_out)
        stratified_kfold_split_and_materialize(
            fruit_out_dir=fruit_out,
            seed=args.seed,
            n_splits=args.n_splits,
            fold_index=args.fold
        )

        # 4) data.yaml 생성
        create_data_yaml(
            dataset_dir=fruit_out,
            yaml_path_override=args.yaml_path_override,
            yaml_train_rel=args.yaml_train_rel,
            yaml_val_rel=args.yaml_val_rel,
            yaml_test_rel=args.yaml_test_rel,
            target_names=target_names
        )

    print("\n=== Done ===")

if __name__ == "__main__":
    main()
