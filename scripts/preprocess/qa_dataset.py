#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

DEFAULT_FRUITS = ["berry", "tomato", "pepper", "korean_melon"]

def parse_args():
    p = argparse.ArgumentParser(
        description="Verify that each camera folder under images has a corresponding labels/<...>/<camera>.json"
    )
    p.add_argument("--images-root", type=Path, default=Path("/home/cat123/crowdworks_new/images"),
                   help="Root path of images directory")
    p.add_argument("--labels-root", type=Path, default=Path("/home/cat123/crowdworks_new/labels"),
                   help="Root path of labels directory")
    p.add_argument("--fruits", type=str, default=",".join(DEFAULT_FRUITS),
                   help=f"Comma-separated fruit names to check. Default: {','.join(DEFAULT_FRUITS)}")
    p.add_argument("--report-csv", type=Path, default=Path("qa_missing_labels.csv"),
                   help="Output CSV path for detailed per-camera report")
    p.add_argument("--check-json-load", action="store_true",
                   help="Also try to open and parse JSON files to ensure they are valid JSON")
    return p.parse_args()

def main():
    args = parse_args()
    fruits = [f.strip() for f in args.fruits.split(",") if f.strip()]

    images_root = args.images_root
    labels_root = args.labels_root

    if not images_root.exists():
        print(f"[ERROR] images root not found: {images_root}", file=sys.stderr)
        sys.exit(2)
    if not labels_root.exists():
        print(f"[WARN] labels root not found: {labels_root} (will report all as missing)", file=sys.stderr)

    # camera_key -> stats
    # camera_key := (fruit, session, date, camera)
    camera_stats = defaultdict(lambda: {"num_images": 0, "label_path": None, "label_exists": False, "json_ok": None})

    # 1) Walk images: /images/<fruit>/<session>/<date>/<camera>/*.png
    for fruit in fruits:
        fruit_dir = images_root / fruit
        if not fruit_dir.exists():
            print(f"[WARN] fruit not found under images: {fruit_dir}")
            continue
        # depth pattern: */*/*/*.png  -> session/date/camera/file.png
        for png in fruit_dir.glob("*/*/*/*.png"):
            try:
                # Expect relative parts: <fruit>/<session>/<date>/<camera>/<file>
                rel = png.relative_to(images_root)
                # rel.parts: [fruit, session, date, camera, file]
                if len(rel.parts) < 5:
                    print(f"[WARN] unexpected image path structure, skipping: {png}")
                    continue
                _, session, date, camera, _ = rel.parts[:5]
                key = (fruit, session, date, camera)
                camera_stats[key]["num_images"] += 1
            except Exception as e:
                print(f"[WARN] failed to process image path {png}: {e}")

    # 2) For every camera key seen in images, compute expected label path and existence
    for (fruit, session, date, camera), stat in camera_stats.items():
        label_path = labels_root / fruit / session / date / f"{camera}.json"
        stat["label_path"] = label_path
        if label_path.exists():
            stat["label_exists"] = True
            if args.check_json_load:
                try:
                    with label_path.open("r", encoding="utf-8") as f:
                        json.load(f)
                    stat["json_ok"] = True
                except Exception as e:
                    stat["json_ok"] = False
                    print(f"[ERROR] JSON parse failed: {label_path} -> {e}")
        else:
            stat["label_exists"] = False
            stat["json_ok"] = None

    # 3) Find orphan labels: labels/<fruit>/<session>/<date>/<camera>.json that do not have images
    orphan_labels = []
    for fruit in fruits:
        labels_fruit_dir = labels_root / fruit
        if not labels_fruit_dir.exists():
            continue
        # depth: */*/*.json  -> session/date/camera.json
        for jpath in labels_fruit_dir.glob("*/*/*.json"):
            try:
                rel = jpath.relative_to(labels_root)
                # rel.parts: [fruit, session, date, camera.json]
                if len(rel.parts) < 4:
                    print(f"[WARN] unexpected label path structure, skipping: {jpath}")
                    continue
                _, session, date, camera_json = rel.parts[:4]
                camera = camera_json[:-5] if camera_json.endswith(".json") else camera_json
                key = (fruit, session, date, camera)
                if key not in camera_stats:
                    orphan_labels.append(jpath)
            except Exception as e:
                print(f"[WARN] failed to process label path {jpath}: {e}")

    # 4) Summaries
    total_camera_groups = len(camera_stats)
    total_images = sum(s["num_images"] for s in camera_stats.values())
    missing = [(k, s) for k, s in camera_stats.items() if not s["label_exists"]]
    present = [(k, s) for k, s in camera_stats.items() if s["label_exists"]]

    invalid_json = []
    if any(s["json_ok"] is not None for s in camera_stats.values()):
        invalid_json = [(k, s) for k, s in camera_stats.items() if s["json_ok"] is False]

    # 5) Write CSV report
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.report_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fruit", "session", "date", "camera", "num_images", "label_path", "label_exists", "json_ok"])
        for (fruit, session, date, camera), s in sorted(camera_stats.items()):
            w.writerow([
                fruit, session, date, camera,
                s["num_images"],
                str(s["label_path"]) if s["label_path"] else "",
                "YES" if s["label_exists"] else "NO",
                "" if s["json_ok"] is None else ("OK" if s["json_ok"] else "FAIL"),
            ])

    # 6) Print human-readable summary
    print("=== QA SUMMARY ===")
    print(f"Images root : {images_root}")
    print(f"Labels root : {labels_root}")
    print(f"Fruits      : {', '.join(fruits)}")
    print(f"Total images: {total_images}")
    print(f"Camera groups (image folders): {total_camera_groups}")
    print(f"Labels present: {len(present)}")
    print(f"Labels missing: {len(missing)}")
    if invalid_json:
        print(f"Invalid JSON files: {len(invalid_json)}")
    print(f"Orphan labels (no images): {len(orphan_labels)}")
    print(f"CSV report: {args.report_csv}")

    if missing:
        print("\n-- Missing label JSONs --")
        for (fruit, session, date, camera), s in sorted(missing):
            print(str(s["label_path"]))

    if orphan_labels:
        print("\n-- Orphan label JSONs (no corresponding images) --")
        for p in sorted(orphan_labels):
            print(str(p))

    # exit code: 0 OK, 1 missing labels, 3 invalid json, 4 both; preference to most severe
    exit_code = 0
    if missing and invalid_json:
        exit_code = 4
    elif invalid_json:
        exit_code = 3
    elif missing:
        exit_code = 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
