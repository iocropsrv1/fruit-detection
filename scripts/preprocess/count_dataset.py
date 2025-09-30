#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

DEFAULT_FRUITS = ["berry", "tomato", "pepper", "korean_melon"]
DEFAULT_SPLITS = ["all", "train", "val", "test"]
VALID_CLASSES = {0, 1, 2}  # 0:ripened, 1:ripening, 2:unripened

def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO dataset QA: count images/objects/class distribution per fruit/split and validate matches."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("/home/cat123/yolov8-fruit_detection/yolo_dataset_CW_new"), # data_path
        help="Dataset root: contains <fruit>/images/<split> and <fruit>/labels/<split>",
    )
    p.add_argument(
        "--fruits",
        type=str,
        default=",".join(DEFAULT_FRUITS),
        help=f"Comma-separated fruits. Default: {','.join(DEFAULT_FRUITS)}",
    )
    p.add_argument(
        "--splits",
        type=str,
        default=",".join(DEFAULT_SPLITS),
        help=f"Comma-separated splits. Default: {','.join(DEFAULT_SPLITS)}",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("yolo_qa_summary.csv"),
        help="Output path for summary CSV (per fruit/split).",
    )
    p.add_argument(
        "--detail-csv",
        type=Path,
        default=None,
        help="Optional: output path for per-file detail CSV.",
    )
    p.add_argument(
        "--strict-invalid-class",
        action="store_true",
        help="Treat invalid class IDs (not in 0/1/2) as failure in exit code.",
    )
    p.add_argument(
        "--max-warn-lines",
        type=int,
        default=30,
        help="Max number of invalid-class lines to print as warnings.",
    )
    return p.parse_args()

def read_label_counts(label_path: Path):
    """
    Return: (num_objects, c0, c1, c2, invalid_lines)
    invalid_lines: list of (line_no, raw_line, parsed_token or None)
    """
    num_objects = 0
    c0 = c1 = c2 = 0
    invalid_lines = []
    try:
        with label_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue  # empty line -> no object
                parts = s.split()
                # YOLO: class cx cy w h ...; we only need first token
                try:
                    cls = int(parts[0])
                except Exception:
                    invalid_lines.append((i, line.rstrip("\n"), None))
                    continue
                if cls not in VALID_CLASSES:
                    invalid_lines.append((i, line.rstrip("\n"), cls))
                    continue
                num_objects += 1
                if cls == 0:
                    c0 += 1
                elif cls == 1:
                    c1 += 1
                else:
                    c2 += 1
    except Exception as e:
        # treat unreadable file as invalid: return sentinel that triggers warnings
        invalid_lines.append((-1, f"<IOError: {e}>", None))
    return num_objects, c0, c1, c2, invalid_lines

def main():
    args = parse_args()
    fruits = [x.strip() for x in args.fruits.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    # results[(fruit, split)] -> dict
    results = defaultdict(lambda: {
        "num_images": 0,
        "num_label_files": 0,
        "objects_total": 0,
        "class0": 0, "class1": 0, "class2": 0,
        "images_wo_labels": 0,  # count
        "labels_wo_images": 0,  # count
    })

    mismatch_details = {
        "images_wo_labels": defaultdict(list),  # (fruit, split) -> [basename...]
        "labels_wo_images": defaultdict(list),
    }
    invalid_class_records = []  # (fruit, split, label_path, line_no, raw, token)

    # Optional per-file detail rows
    perfile_rows = []

    # Walk fruits × splits
    for fruit in fruits:
        for split in splits:
            img_dir = args.root / fruit / "images" / split
            lbl_dir = args.root / fruit / "labels" / split

            if not img_dir.exists():
                print(f"[WARN] images dir not found: {img_dir}")
            if not lbl_dir.exists():
                print(f"[WARN] labels dir not found: {lbl_dir}")

            img_stems = set()
            if img_dir.exists():
                for p in img_dir.glob("*.png"):
                    img_stems.add(p.stem)

            lbl_stems = set()
            label_paths = {}
            if lbl_dir.exists():
                for p in lbl_dir.glob("*.txt"):
                    stem = p.stem
                    lbl_stems.add(stem)
                    label_paths[stem] = p

            # counts
            results[(fruit, split)]["num_images"] = len(img_stems)
            results[(fruit, split)]["num_label_files"] = len(lbl_stems)

            # mismatches
            only_imgs = sorted(img_stems - lbl_stems)
            only_lbls = sorted(lbl_stems - img_stems)
            results[(fruit, split)]["images_wo_labels"] = len(only_imgs)
            results[(fruit, split)]["labels_wo_images"] = len(only_lbls)
            if only_imgs:
                mismatch_details["images_wo_labels"][(fruit, split)].extend(only_imgs)
            if only_lbls:
                mismatch_details["labels_wo_images"][(fruit, split)].extend(only_lbls)

            # aggregate label stats for matched basenames
            matched = sorted(img_stems & lbl_stems)
            for stem in matched:
                lp = label_paths[stem]
                num_obj, c0, c1, c2, bads = read_label_counts(lp)

                # accumulate per file
                results[(fruit, split)]["objects_total"] += num_obj
                results[(fruit, split)]["class0"] += c0
                results[(fruit, split)]["class1"] += c1
                results[(fruit, split)]["class2"] += c2

                if args.detail_csv:
                    perfile_rows.append({
                        "fruit": fruit, "split": split, "basename": stem,
                        "num_objects": num_obj, "class0": c0, "class1": c1, "class2": c2,
                        "label_path": str(lp)
                    })

                if bads:
                    for (ln, raw, tok) in bads:
                        invalid_class_records.append((fruit, split, str(lp), ln, raw, tok))

    # Print summary table
    header = [
        "fruit", "split",
        "images", "label_files",
        "img_wo_lbl", "lbl_wo_img",
        "objects", "class0", "class1", "class2", "invalid_cls_lines"
    ]
    rows = []
    for (fruit, split), stats in sorted(results.items()):
        rows.append([
            fruit, split,
            stats["num_images"], stats["num_label_files"],
            stats["images_wo_labels"], stats["labels_wo_images"],
            stats["objects_total"], stats["class0"], stats["class1"], stats["class2"],
            sum(1 for r in invalid_class_records if r[0] == fruit and r[1] == split)
        ])

    # Pretty print
    colw = [max(len(str(x)) for x in col) for col in zip(header, *rows)] if rows else [len(h) for h in header]
    fmt = "  ".join("{:<" + str(w) + "}" for w in colw)
    print("\n=== YOLO DATASET QA SUMMARY ===")
    print(f"Root: {args.root}")
    print(fmt.format(*header))
    print("-" * (sum(colw) + 2 * (len(colw) - 1)))
    for r in rows:
        print(fmt.format(*r))

    # Write summary CSV
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    print(f"\n[INFO] Summary CSV: {args.summary_csv}")

    # Write detail CSV
    if args.detail_csv:
        args.detail_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.detail_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["fruit","split","basename","num_objects","class0","class1","class2","label_path"])
            w.writeheader()
            for row in perfile_rows:
                w.writerow(row)
        print(f"[INFO] Detail CSV: {args.detail_csv}")

    # Print mismatches (limited)
    if mismatch_details["images_wo_labels"] or mismatch_details["labels_wo_images"]:
        print("\n-- MISMATCH DETAILS --")
        for key, stems in sorted(mismatch_details["images_wo_labels"].items()):
            if stems:
                fruit, split = key
                print(f"[images_wo_labels] {fruit}/{split}: {len(stems)}")
                # print a few examples
                for s in stems[:20]:
                    print(f"  - {s}.png (label missing)")
                if len(stems) > 20:
                    print(f"  (+{len(stems)-20} more)")
        for key, stems in sorted(mismatch_details["labels_wo_images"].items()):
            if stems:
                fruit, split = key
                print(f"[labels_wo_images] {fruit}/{split}: {len(stems)}")
                for s in stems[:20]:
                    print(f"  - {s}.txt (image missing)")
                if len(stems) > 20:
                    print(f"  (+{len(stems)-20} more)")

    # Print invalid class warnings
    if invalid_class_records:
        print("\n-- INVALID CLASS LINES (not in 0/1/2) --")
        for idx, (fruit, split, lp, ln, raw, tok) in enumerate(invalid_class_records[:args.max_warn_lines], start=1):
            print(f"[{idx}] {fruit}/{split} {lp}:{ln} -> token={tok} line='{raw}'")
        if len(invalid_class_records) > args.max_warn_lines:
            print(f"... and {len(invalid_class_records) - args.max_warn_lines} more")

    # Exit code policy
    has_mismatch = any(
        stats["images_wo_labels"] or stats["labels_wo_images"]
        for stats in results.values()
    )
    has_invalid = len(invalid_class_records) > 0

    exit_code = 0
    if has_mismatch and (args.strict_invalid_class and has_invalid):
        exit_code = 3  # both
    elif has_mismatch:
        exit_code = 1
    elif args.strict_invalid_class and has_invalid:
        exit_code = 2

    if exit_code == 0:
        print("\n[OK] No mismatches. No invalid classes (or strict flag not set).")
    elif exit_code == 1:
        print("\n[FAIL] Mismatches found (images↔labels).")
    elif exit_code == 2:
        print("\n[FAIL] Invalid class IDs found.")
    else:
        print("\n[FAIL] Mismatches and invalid class IDs found.")

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
