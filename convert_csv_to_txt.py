#!/usr/bin/env python3
import csv
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

try:
    from PIL import Image
except ImportError:
    Image = None  # --width/--height를 쓰면 PIL 없이도 동작

# 입력 경로 설정
CSV_PATH   = "/home/cat123/eval_data/labels/berry/berry_20230529_182424/berry_1_batch_000.csv"
IMAGES_DIR = "/home/cat123/eval_data/dataset/images/berry/berry_1_batch_000"  
LABELS_DIR = "/home/cat123/eval_data/dataset/labels/berry/berry_1_batch_000"

# (선택) 모든 이미지의 해상도가 동일하고 알고 있다면 지정하여 PIL 의존 제거
# 둘 다 None이면 실제 이미지를 열어 W,H를 읽습니다.
FIXED_WIDTH: Optional[int]  = None   # 예: 1280
FIXED_HEIGHT: Optional[int] = None   # 예: 960

# 허용 이미지 확장자
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

os.makedirs(LABELS_DIR, exist_ok=True)

def build_frame_to_stem(images_dir: Path) -> Tuple[Dict[int, str], Dict[str, Path]]:
    """
    이미지 디렉터리에서 파일을 정렬하여
    frame_id(0,1,2,...) -> 파일 stem 매핑과 stem -> 전체 경로 매핑을 만든다.
    """
    files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    stems = [p.stem for p in files]
    frame_to_stem = {i: stems[i] for i in range(len(stems))}
    stem_to_path = {p.stem: p for p in files}
    if not files:
        raise FileNotFoundError(f"이미지 파일이 없습니다: {images_dir}")
    return frame_to_stem, stem_to_path

def get_image_size(stem: str, stem_to_path: Dict[str, Path]) -> Tuple[int, int]:
    """
    (W,H) 반환.
    - FIXED_WIDTH/HEIGHT 지정 시 그대로 사용
    - 아니면 실제 이미지를 열어서 읽음
    """
    if FIXED_WIDTH is not None and FIXED_HEIGHT is not None:
        return FIXED_WIDTH, FIXED_HEIGHT
    if Image is None:
        raise RuntimeError("PIL이 없고 FIXED_WIDTH/HEIGHT도 지정되지 않았습니다.")
    img_path = stem_to_path[stem]
    with Image.open(img_path) as im:
        return im.size  # (W, H)

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def main():
    images_dir = Path(IMAGES_DIR)
    labels_dir = Path(LABELS_DIR)

    frame_to_stem, stem_to_path = build_frame_to_stem(images_dir)

    n_rows = n_written = n_files = 0
    current_stem = None
    out_fp = None
    W = H = None  # 현재 stem의 이미지 크기

    try:
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_rows += 1
                # CSV 스키마: frame_id,track_id,x1,y1,x2,y2,class
                fid = int(float(row["frame_id"]))  # "0" 또는 "0.0" 모두 대응
                cls = int(float(row["class"]))
                x1  = float(row["x1"]); y1 = float(row["y1"])
                x2  = float(row["x2"]); y2 = float(row["y2"])

                # frame_id -> 실제 파일 stem 매핑
                if fid not in frame_to_stem:
                    # CSV에 frame_id가 이미지 개수보다 클 수 있음 → 스킵
                    continue
                stem = frame_to_stem[fid]
                label_path = labels_dir / f"{stem}.txt"

                # 프레임(=파일) 전환 시 핸들/사이즈 갱신
                if current_stem != stem:
                    if out_fp is not None:
                        out_fp.close()
                    current_stem = stem
                    out_fp = open(label_path, "w", encoding="utf-8")
                    n_files += 1
                    W, H = get_image_size(stem, stem_to_path)

                # 방어적 클램프 후 YOLO 정규화 cx cy w h
                x1c = max(0.0, min(x1, W)); x2c = max(0.0, min(x2, W))
                y1c = max(0.0, min(y1, H)); y2c = max(0.0, min(y2, H))
                if x2c <= x1c or y2c <= y1c:
                    # 비정상 박스 스킵
                    continue

                cx = ((x1c + x2c) / 2.0) / W
                cy = ((y1c + y2c) / 2.0) / H
                ww = (x2c - x1c) / W
                hh = (y2c - y1c) / H

                cx = clamp01(cx); cy = clamp01(cy); ww = clamp01(ww); hh = clamp01(hh)

                # YOLO 포맷: class cx cy w h  (공백 구분, 정규화)
                out_fp.write(f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}\n")
                n_written += 1
    finally:
        if out_fp is not None:
            out_fp.close()

    print(f"rows read: {n_rows}, labels written: {n_written}, files: {n_files}")
    print(f"labels dir: {labels_dir}")

if __name__ == "__main__":
    main()
