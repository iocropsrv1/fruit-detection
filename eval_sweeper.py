#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실행 예시:
  python eval_sweeper.py \
    --eval_script evaluate_model.py \
    --models 모델경로 \
    --datasets data/yolo_dataset_v1.0/berry_v1.0.yaml,data/yolo_dataset_v1.1/berry_v1.1.yaml \
    --fruits pepper,tomato \
    --output_dir outputs \
    --conf_thresholds 0.25,0.3 \
    --iou_thresholds 0.45 \
    --samples 20 \
    --max_concurrent 2 \
    --gpus 0,1 \
    --extra "--test_images_dir ''"

설명:
  - models, datasets는 콤마(,)로 구분하여 여러 개를 전달
  - fruits는 1개만 주면 모든 dataset에 동일 적용, 여러 개를 주면 datasets 갯수와 동일해야 함
  - 각 조합은 고유 서브폴더로 평가되고, 최종 요약은 <output_dir>/evaluation_sweep_results.csv 에 집계됨
  - 클래스별 mAP50 (map50_ripened, map50_ripening, map50_unripened) 포함
  - 실패(OOM 등)해도 다음 조합으로 계속 진행
"""

import argparse
import csv
import os
import random
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


# =========================
# 데이터 구조
# =========================
@dataclass
class EvalJob:
    model_path: str
    data_yaml: str
    fruit: str
    conf: float
    iou: float
    samples: int
    output_base: str
    run_dir: Optional[str] = None
    gpu_id: Optional[str] = None
    run_name: Optional[str] = None
    extra_args: str = ""


# =========================
# 파서 & 유틸
# =========================
def parse_list(s: Optional[str], cast):
    if not s:
        return []
    return [cast(x.strip()) for x in s.split(",") if x.strip() != ""]


def build_parser():
    p = argparse.ArgumentParser(description="YOLOv8 evaluation sweeper")
    # 필수
    p.add_argument("--eval_script", required=True, help="evaluate_model.py 경로")
    p.add_argument("--models", required=True, help="모델 .pt 경로들(콤마 구분)")
    p.add_argument("--datasets", required=True, help="test data.yaml 경로들(콤마 구분)")
    p.add_argument("--output_dir", required=True, help="스위프 결과 최상위 출력 디렉토리")

    # 과실명 매핑
    p.add_argument("--fruits", default="", help="과실명(1개 또는 datasets 길이와 동일한 개수)")

    # 평가 하이퍼파라미터 sweep 축
    p.add_argument("--conf_thresholds", default="0.25", help="콤마 구분 (예: 0.25,0.3)")
    p.add_argument("--iou_thresholds", default="0.45", help="콤마 구분 (예: 0.45)")
    p.add_argument("--samples", default="20", help="시각화 샘플 수(콤마 구분 가능)")

    # 실행 제어
    p.add_argument("--max_concurrent", type=int, default=1, help="동시 실행 개수")
    p.add_argument("--gpus", default="", help="라운드로빈 GPU IDs (예: 0,1)")
    p.add_argument("--dry_run", action="store_true", help="실제 실행하지 않고 커맨드만 출력")
    p.add_argument("--extra", default="", help="evaluate_model.py에 그대로 전달할 추가 인자들")

    # 선택: CUDA allocator 튜닝(메모리 단편화 방지 도움)
    p.add_argument("--cuda_alloc_conf", default="", help='예: "max_split_size_mb:128,expandable_segments:True"')

    return p


def validate_fruits(datasets: List[str], fruits: List[str]) -> List[str]:
    if not fruits:
        # 과실명을 dataset stem으로 자동 설정
        return [Path(ds).stem for ds in datasets]
    if len(fruits) == 1:
        return [fruits[0] for _ in datasets]
    if len(fruits) != len(datasets):
        raise ValueError("--fruits 개수는 1개이거나 --datasets 개수와 동일해야 합니다.")
    return fruits


def make_run_name(model_path: str, data_yaml: str, fruit: str, conf: float, iou: float, samples: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    mstem = Path(model_path).stem
    dstem = Path(data_yaml).stem
    return f"eval_{ts}_m={mstem}_ds={dstem}_f={fruit}_conf={conf}_iou={iou}_s={samples}"


def build_cmd(args, job: EvalJob) -> List[str]:
    cmd = [
        sys.executable, args.eval_script,
        "--model", job.model_path,
        "--data", job.data_yaml,
        "--output_dir", job.run_dir,
        "--fruit", job.fruit,
        "--conf_threshold", str(job.conf),
        "--iou_threshold", str(job.iou),
    ]
    if args.extra:
        cmd += shlex.split(args.extra)
    return cmd


def get_class_names_from_data_yaml(data_yaml: str) -> List[str]:
    """data.yaml에서 클래스 이름들을 추출"""
    try:
        with open(data_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('names', [])
    except:
        return []


def ensure_summary_header(csv_path: Path, all_class_names: List[str]) -> List[str]:
    """클래스별 mAP50 컬럼을 포함한 헤더 생성"""
    header = [
        "idx",
        "status",
        "returncode",
        "duration_sec",
        "gpu",
        "run_dir",
        "model",
        "dataset",
        "fruit",
        "conf",
        "iou",
        "samples",
        # 전체 metrics
        "map50",
        "map50_95",
        "precision",
        "recall",
        "f1_score",
    ]
    
    # 클래스별 mAP50 컬럼 추가
    for class_name in all_class_names:
        header.append(f"map50_{class_name.lower()}")
    
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
    return header


def append_summary(csv_path: Path, header: List[str], row: Dict[str, Any]):
    full = {k: row.get(k, "") for k in header}
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writerow(full)


def extract_comprehensive_metrics_from_run(run_dir: Path) -> Dict[str, Any]:
    """
    수정된 evaluate_model.py가 저장한 evaluation_results*.csv에서
    전체 지표와 클래스별 mAP50을 모두 추출
    """
    metrics = {
        "map50": "", "map50_95": "", "precision": "", "recall": "", "f1_score": ""
    }

    csvs = sorted(run_dir.glob("evaluation_results*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        return metrics

    try:
        df = pd.read_csv(csvs[0])
        if not df.empty:
            row = df.iloc[0]  # 첫 번째 행 사용
            
            # 전체 메트릭
            metrics["map50"] = row.get("map50", "")
            metrics["map50_95"] = row.get("map50_95", "")
            metrics["precision"] = row.get("precision", "")
            metrics["recall"] = row.get("recall", "")
            metrics["f1_score"] = row.get("f1_score", "")
            
            # 클래스별 mAP50 추출 (map50_로 시작하는 컬럼들)
            for col in df.columns:
                if col.startswith("map50_") and col != "map50_95":
                    class_name = col.replace("map50_", "")
                    metrics[f"map50_{class_name}"] = row.get(col, "")
                    
    except Exception as e:
        print(f"⚠️ 메트릭 파싱 오류: {e}")

    return metrics


# =========================
# 실행 단위
# =========================
def run_one(idx: int, args, job: EvalJob, gpu_cycle: List[str], summary_csv: Path, header: List[str]) -> Dict[str, Any]:
    env = os.environ.copy()
    if gpu_cycle:
        job.gpu_id = gpu_cycle[idx % len(gpu_cycle)]
        env["CUDA_VISIBLE_DEVICES"] = job.gpu_id
    if args.cuda_alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf

    # 고유 run_dir 생성
    job.run_name = make_run_name(job.model_path, job.data_yaml, job.fruit, job.conf, job.iou, job.samples)
    job.run_dir = str(Path(job.output_base) / job.run_name)
    Path(job.run_dir).mkdir(parents=True, exist_ok=True)

    cmd = build_cmd(args, job)

    print(f"\n=== [{idx+1}] EVAL {job.run_name} ===")
    if job.gpu_id:
        print(f"GPU -> {job.gpu_id}")
    print("CMD:", " ".join(shlex.quote(c) for c in cmd))

    start = time.time()
    status = "SKIPPED" if args.dry_run else "OK"
    rc = 0

    if not args.dry_run:
        proc = subprocess.run(cmd, env=env, check=False)
        rc = proc.returncode
        status = "OK" if rc == 0 else "FAIL"

    dur = time.time() - start

    # 메트릭 수집(가능한 경우)
    metrics = {"map50": "", "map50_95": "", "precision": "", "recall": "", "f1_score": ""}
    if status == "OK":
        metrics = extract_comprehensive_metrics_from_run(Path(job.run_dir))

    row = {
        "idx": idx + 1,
        "status": status,
        "returncode": rc,
        "duration_sec": f"{dur:.1f}",
        "gpu": job.gpu_id or "",
        "run_dir": job.run_dir,
        "model": job.model_path,
        "dataset": job.data_yaml,
        "fruit": job.fruit,
        "conf": job.conf,
        "iou": job.iou,
        "samples": job.samples,
    }
    
    # 전체 지표와 클래스별 지표 모두 추가
    row.update(metrics)
    
    append_summary(summary_csv, header, row)
    return row


# =========================
# 메인
# =========================
def main():
    parser = build_parser()
    args = parser.parse_args()

    models = parse_list(args.models, str)
    datasets = parse_list(args.datasets, str)
    confs = parse_list(args.conf_thresholds, float)
    ious = parse_list(args.iou_thresholds, float)
    samples_list = parse_list(args.samples, int)
    fruits_in = parse_list(args.fruits, str)

    if not models or not datasets or not confs or not ious or not samples_list:
        raise ValueError("models, datasets, conf_thresholds, iou_thresholds, samples는 최소 1개 이상 필요합니다.")

    fruits = validate_fruits(datasets, fruits_in)

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 모든 데이터셋에서 클래스 이름 수집
    all_class_names = set()
    for dataset in datasets:
        class_names = get_class_names_from_data_yaml(dataset)
        all_class_names.update(class_names)
    
    all_class_names = sorted(list(all_class_names))  # 정렬된 리스트로 변환
    
    summary_csv = output_base / "evaluation_sweep_results.csv"
    header = ensure_summary_header(summary_csv, all_class_names)

    # 조합 생성: 모델 × 데이터셋 × (conf × iou × samples)
    jobs: List[EvalJob] = []
    for model_path, data_yaml in product(models, datasets):
        fruit = fruits[datasets.index(data_yaml)]
        for conf, iou, smp in product(confs, ious, samples_list):
            jobs.append(EvalJob(
                model_path=model_path,
                data_yaml=data_yaml,
                fruit=fruit,
                conf=conf,
                iou=iou,
                samples=smp,
                output_base=str(output_base),
                extra_args=args.extra
            ))

    print(f"Total eval jobs: {len(jobs)}")
    print(f"클래스: {all_class_names}")

    gpu_cycle = parse_list(args.gpus, str)
    if gpu_cycle:
        print(f"Using GPUs (round-robin): {gpu_cycle}")

    if args.dry_run:
        for i, j in enumerate(jobs):
            if gpu_cycle:
                j.gpu_id = gpu_cycle[i % len(gpu_cycle)]
            # preview run_dir
            rn = make_run_name(j.model_path, j.data_yaml, j.fruit, j.conf, j.iou, j.samples)
            rd = str(output_base / rn)
            cmd = build_cmd(args, EvalJob(**{**j.__dict__, "run_dir": rd}))
            print(f"[DRY-RUN {i+1}] GPU={j.gpu_id or '-'} CMD:", " ".join(shlex.quote(c) for c in cmd))
        print("Dry-run complete.")
        return

    max_workers = max(1, int(args.max_concurrent))
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(run_one, i, args, job, gpu_cycle, summary_csv, header)
            for i, job in enumerate(jobs)
        ]
        for fut in as_completed(futures):
            res = fut.result()
            if res["status"] == "OK":
                ok += 1
            else:
                fail += 1
            print(f"[Progress] OK={ok} FAIL={fail} / {len(jobs)}")

    print(f"\nSweep finished. Summary CSV: {summary_csv}")
    
    # 결과 요약 출력
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            successful_runs = df[df['status'] == 'OK']
            if not successful_runs.empty:
                print(f"\n=== 성공한 실험 결과 요약 ===")
                print(f"성공: {len(successful_runs)}/{len(df)} 실험")
                if 'map50' in successful_runs.columns:
                    best_result = successful_runs.loc[successful_runs['map50'].idxmax()]
                    print(f"최고 mAP50: {best_result['map50']:.4f}")
                    print(f"  - 모델: {Path(best_result['model']).name}")
                    print(f"  - 과실: {best_result['fruit']}")
                    print(f"  - 설정: conf={best_result['conf']}, iou={best_result['iou']}")
        except:
            pass


if __name__ == "__main__":
    main()