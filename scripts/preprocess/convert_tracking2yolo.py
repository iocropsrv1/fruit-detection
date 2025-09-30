#!/usr/bin/env python3
import json, argparse
from pathlib import Path
from collections import defaultdict

def clamp01(v): 
    return max(0.0, min(1.0, v))

def to_yolo_xywhn(box, img_w, img_h):
    x, y, w, h = box  # COCO: top-left x,y,width,height (픽셀)
    x_c = (x + w/2.0) / img_w
    y_c = (y + h/2.0) / img_h
    w_n = w / img_w
    h_n = h / img_h
    # 혹시 박스가 프레임 밖으로 나간 경우를 대비해 살짝 클램프
    return tuple(clamp01(v) for v in (x_c, y_c, w_n, h_n))

def main():
    ap = argparse.ArgumentParser(description="COCO tracking JSON → YOLO (det/track)")
    ap.add_argument("--json", required=True, help="COCO tracking json 경로")
    ap.add_argument("--out",  required=True, help="출력 라벨 디렉토리 (labels)")
    ap.add_argument("--format", choices=["det","track"], default="det", help="라벨 포맷")
    ap.add_argument("--names", default=None, help="(선택) classes.names 또는 data.yaml용 names만 출력")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 이미지 메타: id → (file_name, w, h)
    img_meta = {}
    for im in coco["images"]:
        img_meta[im["id"]] = (im["file_name"], im["width"], im["height"])

    # 카테고리: COCO id(1..N) → YOLO class(0..N-1)
    cats_sorted = sorted(coco["categories"], key=lambda c: c["id"])
    id2yolo = {c["id"]: i for i,c in enumerate(cats_sorted)}
    yolo_names = [c["name"] for c in cats_sorted]

    # 이미지별 어노테이션 묶기
    by_img = defaultdict(list)
    for ann in coco["annotations"]:
        by_img[ann["image_id"]].append(ann)

    # 변환 & 저장
    count_labels = 0
    for img_id, anns in by_img.items():
        file_name, W, H = img_meta[img_id]
        stem = Path(file_name).stem
        label_path = out_dir / f"{stem}.txt"

        lines = []
        for ann in anns:
            cls = id2yolo.get(ann["category_id"], None)
            if cls is None: 
                continue
            x_c, y_c, w_n, h_n = to_yolo_xywhn(ann["bbox"], W, H)

            if args.format == "det":
                line = f"{cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
            else:  # "track"
                tid = ann.get("tracking_id", -1)
                # YOLOv8 tracking은 정수 track_id 권장
                tid = int(tid) if isinstance(tid, (int, float)) else -1
                line = f"{cls} {tid} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"

            lines.append(line)

        if lines:
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            count_labels += 1

    # names 출력 옵션
    if args.names:
        names_path = Path(args.names)
        if names_path.suffix.lower() in {".yaml", ".yml"}:
            yaml_txt = "nc: {}\nnames: {}\n".format(len(yolo_names), yolo_names)
            names_path.write_text(yaml_txt, encoding="utf-8")
        else:
            names_path.write_text("\n".join(yolo_names) + "\n", encoding="utf-8")

    print(f"Done. Wrote {count_labels} label files to: {out_dir}")
    print(f"Classes (YOLO index order): {yolo_names}")

if __name__ == "__main__":
    main()
