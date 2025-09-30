#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare YOLO Dataset from Crowdworks (COCO JSON → YOLO txt)
- Converts COCO(JSON) labels to YOLO format
- Organizes by fruit type into staging (images/all, labels/all)
- Splits dataset using Hold-Out method: 80% train, 10% val, 10% test
- Generates YOLO-compatible `data.yaml` for each fruit

Output structure:
  <output_dir>/<fruit>/
      images/{train,val,test}/
      labels/{train,val,test}/
      data.yaml
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Default class names
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
    """Normalize class names with mapping rules."""
    return TARGET_NAME_MAP.get(
        name.strip().lower().replace(' ', '').replace('\t', ''),
        name.strip().lower()
    )


def convert_bbox_coco_to_yolo(bbox, img_w, img_h):
    """Convert COCO format bbox [x,y,w,h] → YOLO format [cx,cy,w,h]."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return [cx, cy, w / img_w, h / img_h]


def ensure_dirs_for_fruit(base: Path):
    """Create directory structure for images/labels."""
    (base / 'images' / 'all').mkdir(parents=True, exist_ok=True)
    (base / 'labels' / 'all').mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        (base / 'images' / split).mkdir(parents=True, exist_ok=True)
        (base / 'labels' / split).mkdir(parents=True, exist_ok=True)


def create_data_yaml(dataset_dir: Path,
                     yaml_path_override: str,
                     yaml_train_rel: str,
                     yaml_val_rel: str,
                     yaml_test_rel: str,
                     target_names):
    """Write YOLO-style data.yaml file."""
    content = (
        f"path: {yaml_path_override}\n"
        f"train: {yaml_train_rel}\n"
        f"val: {yaml_val_rel}\n"
        f"test: {yaml_test_rel}\n\n"
        f"nc: {len(target_names)}\n"
        f"names: {target_names}\n"
    )
    yaml_path = dataset_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"[data.yaml] written → {yaml_path}")


def collect_items(input_root: Path):
    """
    Collect dataset items.

    Returns:
        dict: items_by_fruit[fruit] = list of dicts(
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


def convert_items_to_staging_all(items, fruit_out_dir: Path, target_names, strict_drop_unknown=True):
    """
    Convert COCO → YOLO labels and store into staging (all).
    Keeps empty label files if no boxes found (YOLO allows empty).
    """
    ensure_dirs_for_fruit(fruit_out_dir)
    class_map = {name: idx for idx, name in enumerate(target_names)}
    coco_cache = {}
    kept, dropped = 0, 0

    for info in items:
        img_path = info['image_path']
        json_path = info['json_path']

        if json_path not in coco_cache:
            with open(json_path, 'r', encoding='utf-8') as f:
                coco_cache[json_path] = json.load(f)
        coco = coco_cache[json_path]

        # Index COCO
        images_info = {img['id']: img for img in coco.get('images', [])}
        anns_by_image = defaultdict(list)
        for ann in coco.get('annotations', []):
            anns_by_image[ann['image_id']].append(ann)

        cat_name_by_id = {
            cat['id']: normalize_class_name(cat.get('name', ''))
            for cat in coco.get('categories', [])
        }

        matched_img, matched_anns = None, []
        for img_id, meta in images_info.items():
            if meta.get('file_name') == info['file_name']:
                matched_img, matched_anns = meta, anns_by_image[img_id]
                break
        if not matched_img:
            print(f"[Warn] no annotation for {info['file_name']}")
            continue

        # Destination file names
        base_name = f"{info['fruit']}_{info['region']}_{info['date']}_{info['section']}_{info['file_name']}"
        dst_img = fruit_out_dir / 'images' / 'all' / base_name
        shutil.copy2(img_path, dst_img)

        dst_lbl = fruit_out_dir / 'labels' / 'all' / (dst_img.stem + '.txt')

        # Write YOLO label
        count_lines = 0
        with open(dst_lbl, 'w', encoding='utf-8') as lf:
            for ann in matched_anns:
                if 'category_id' not in ann or 'bbox' not in ann:
                    continue
                cname = cat_name_by_id.get(ann['category_id'])
                if cname not in class_map:
                    if strict_drop_unknown:
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

    print(f"[Staging] {fruit_out_dir.name}: kept(>=1 box)={kept}, empty={dropped}")


def random_split_80_10_10_and_materialize(fruit_out_dir: Path, seed: int):
    """
    Perform random 80/10/10 split and copy files to train/val/test.
    """
    rng = random.Random(seed)

    all_imgs = sorted((fruit_out_dir / 'images' / 'all').glob('*.png'))
    all_basenames = [p.stem for p in all_imgs]

    if len(all_basenames) == 0:
        print(f"[Split] {fruit_out_dir.name}: nothing in staging.")
        return

    # Split
    names_rest, names_test = train_test_split(all_basenames, test_size=0.1, random_state=seed)
    names_train, names_val = train_test_split(names_rest, test_size=(0.1 / 0.9), random_state=seed)

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
    copy_for_split('val', names_val)
    copy_for_split('test', names_test)


def main():
    parser = argparse.ArgumentParser(description="Convert Crowdworks COCO dataset → YOLO with Holdout split")
    parser.add_argument('--input_dir', required=True, help='Input root (with images/, labels/)')
    parser.add_argument('--output_dir', required=True, help='Output root (per fruit subdir will be created)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--yaml_path_override', default='/data/ioCrops/berry/dataset/fruit/train_v1.0')
    parser.add_argument('--yaml_train_rel', default='images/train/')
    parser.add_argument('--yaml_val_rel', default='images/test/')  # original request
    parser.add_argument('--yaml_test_rel', default='images/test/')
    parser.add_argument('--yaml_names', default='ripened,ripening,unripened')
    args = parser.parse_args()

    random.seed(args.seed)

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    target_names = [s.strip() for s in args.yaml_names.split(',') if s.strip()] or TARGET_NAMES_DEFAULT

    # Collect
    items_by_fruit = collect_items(input_root)

    # Convert
    for fruit, items in sorted(items_by_fruit.items()):
        fruit_out = output_root / fruit
        print(f"\n=== Convert to YOLO (staging) | fruit={fruit}, N={len(items)} ===")
        convert_items_to_staging_all(items, fruit_out, target_names, strict_drop_unknown=True)

    # Split & materialize
    for fruit in sorted(items_by_fruit.keys()):
        fruit_out = output_root / fruit
        print(f"\n=== Random split 0.8/0.1/0.1 | fruit={fruit} ===")
        ensure_dirs_for_fruit(fruit_out)
        random_split_80_10_10_and_materialize(fruit_out, seed=args.seed)

        # Write data.yaml
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
