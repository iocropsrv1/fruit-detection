#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실행 예시:
  python train_sweeper.py \
    --train_script train_yolov8_mixparams.py \
    --datasets data/yolo_dataset_v1.0/berry_v1.0.yaml,data/yolo_dataset_v1.1/berry_v1.1.yaml \
    --models s,m \
    --epochs 5,10,20 \
    --batch_size 16,32 \
    --optimizer SGD,Adam \
    --lr0 0.01,0.001 \
    --imgsz 640,1080 \
    --output_dir outputs \
    --max_concurrent 2 \
    --gpus 0,1 \
    --prefix exp \
    --fruit_from_dataset \
    --extra "--workers 8"

설명:
  - Detection에서 유효한 인자만 sweep 축으로 노출.
  - CUDA OOM 등 실패 시 CSV에 FAIL로 기록 후 다음 조합으로 진행.
  - 수정된 train_yolov8_mixparams.py와 호환 (직관적 디렉토리명)
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
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Any

@dataclass
class Job:
    dataset: str
    model: str
    epochs: int
    sweep_args: Dict[str, Any]
    fixed_args: Dict[str, Any]
    output_dir: str
    run_name: str = ""
    gpu_id: str = ""

def parse_list_arg(arg_val: str, cast_fn):
    if arg_val is None or arg_val == "":
        return []
    return [cast_fn(x.strip()) for x in arg_val.split(",") if x.strip() != ""]

def build_arg_parser():
    p = argparse.ArgumentParser(description="Robust grid sweep runner for YOLOv8 training (Detection)")
    # 필수
    p.add_argument("--train_script", required=True)
    p.add_argument("--output_dir", required=True)
    # 축
    p.add_argument("--datasets", required=True)
    p.add_argument("--models", required=True)   # n,s,m,l,x
    p.add_argument("--epochs", required=True)   # ex) 5,10,20
    # 주요 하이퍼파라미터
    p.add_argument("--optimizer", default="")  # SGD,Adam,AdamW 추가
    p.add_argument("--lr0", default="")
    p.add_argument("--batch_size", default="")
    p.add_argument("--imgsz", default="")
    # Detection 유효 Augs
    p.add_argument("--mosaic", default="")
    p.add_argument("--mixup", default="")
    p.add_argument("--copy_paste", default="")
    p.add_argument("--hsv_h", default="")
    p.add_argument("--hsv_s", default="")
    p.add_argument("--hsv_v", default="")
    p.add_argument("--fliplr", default="")
    p.add_argument("--flipud", default="")
    p.add_argument("--degrees", default="")
    p.add_argument("--translate", default="")
    p.add_argument("--scale", default="")
    p.add_argument("--shear", default="")
    p.add_argument("--perspective", default="")
    p.add_argument("--close_mosaic", default="")
    # 기타
    p.add_argument("--freeze", default="")
    p.add_argument("--cos_lr", action="store_true")
    p.add_argument("--no_cos_lr", action="store_true")
    # 실행 제어
    p.add_argument("--max_concurrent", type=int, default=1)
    p.add_argument("--gpus", default="")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--sample_k", type=int, default=0)
    p.add_argument("--extra", default="")
    # 네이밍
    p.add_argument("--prefix", default="sweep")
    p.add_argument("--fruit_from_dataset", action="store_true")
    # 선택: CUDA allocator 튜닝
    p.add_argument("--cuda_alloc_conf", default="")
    return p

def expand_grid(args) -> Tuple[List[Job], List[str]]:
    datasets = parse_list_arg(args.datasets, str)
    models = parse_list_arg(args.models, str)
    epochs_list = parse_list_arg(args.epochs, int)

    sweep_spaces: Dict[str, List[Any]] = {}
    def add_space(key, raw, caster):
        vals = parse_list_arg(raw, caster)
        if vals:
            sweep_spaces[key] = vals

    # 주요 하이퍼파라미터
    add_space("optimizer", args.optimizer, str)  # 추가
    add_space("lr0", args.lr0, float)
    add_space("batch_size", args.batch_size, int)
    add_space("imgsz", args.imgsz, int)
    # Detection augs
    add_space("mosaic", args.mosaic, float)
    add_space("mixup", args.mixup, float)
    add_space("copy_paste", args.copy_paste, float)
    add_space("hsv_h", args.hsv_h, float)
    add_space("hsv_s", args.hsv_s, float)
    add_space("hsv_v", args.hsv_v, float)
    add_space("fliplr", args.fliplr, float)
    add_space("flipud", args.flipud, float)
    add_space("degrees", args.degrees, float)
    add_space("translate", args.translate, float)
    add_space("scale", args.scale, float)
    add_space("shear", args.shear, float)
    add_space("perspective", args.perspective, float)
    add_space("close_mosaic", args.close_mosaic, int)
    # 기타
    add_space("freeze", args.freeze, int)

    fixed_args: Dict[str, Any] = {}
    if args.cos_lr and not args.no_cos_lr:
        fixed_args["cos_lr"] = True
    if args.no_cos_lr and not args.cos_lr:
        fixed_args["cos_lr"] = False

    sweep_keys = list(sweep_spaces.keys())
    sweep_values = [sweep_spaces[k] for k in sweep_keys]

    jobs: List[Job] = []
    if sweep_keys:
        for dataset, epochs, model, combo in product(datasets, epochs_list, models, product(*sweep_values)):
            sweep_args = dict(zip(sweep_keys, combo))
            jobs.append(Job(dataset, model, epochs, sweep_args, fixed_args, args.output_dir))
    else:
        for dataset, epochs, model in product(datasets, epochs_list, models):
            jobs.append(Job(dataset, model, epochs, {}, fixed_args, args.output_dir))

    if args.sample_k and 0 < args.sample_k < len(jobs):
        jobs = random.sample(jobs, args.sample_k)

    for j in jobs:
        ds_stem = Path(j.dataset).stem
        parts = [args.prefix, f"ds={ds_stem}", f"m={j.model}", f"e={j.epochs}"]
        for k in sorted(j.sweep_args.keys()):
            parts.append(f"{k}={j.sweep_args[k]}")
        j.run_name = "_".join(parts)
        j.fixed_args["fruit"] = (ds_stem if args.fruit_from_dataset else j.run_name)

    return jobs, sweep_keys

def build_cmd(args, job: Job) -> List[str]:
    cmd = [
        sys.executable, args.train_script,
        "--data", job.dataset,
        "--output_dir", job.output_dir,
        "--model_size", job.model,
        "--epochs", str(job.epochs),
        "--fruit", job.fixed_args.get("fruit", job.run_name),
    ]
    
    # cos_lr 처리
    if "cos_lr" in job.fixed_args:
        if job.fixed_args["cos_lr"]:
            cmd += ["--cos_lr"]

    # sweep 파라미터들 추가
    for k, v in job.sweep_args.items():
        cmd += [f"--{k}", str(v)]

    # 추가 인자들
    if args.extra:
        cmd += shlex.split(args.extra)

    return cmd

def ensure_csv_header(csv_path: Path, sweep_keys: List[str]):
    base_cols = [
        "idx", "dataset", "model", "epochs", "gpu",
        "run_name", "status", "returncode", "duration_sec", "run_dir"
    ]
    header = base_cols + sorted(sweep_keys)
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
    return header

def append_csv_row(csv_path: Path, header: List[str], row: Dict[str, Any]):
    full_row = {k: row.get(k, "") for k in header}
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writerow(full_row)

def run_one(idx: int, job: Job, args, gpu_cycle: List[str], csv_path: Path, header: List[str]) -> Dict[str, Any]:
    env = os.environ.copy()
    if gpu_cycle:
        job.gpu_id = gpu_cycle[idx % len(gpu_cycle)]
        env["CUDA_VISIBLE_DEVICES"] = job.gpu_id
    if args.cuda_alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf

    cmd = build_cmd(args, job)
    start = time.time()
    status = "SKIPPED" if args.dry_run else "OK"
    rc = 0

    print(f"\n=== [{idx+1}] RUN {job.run_name} ===")
    if job.gpu_id:
        print(f"GPU -> {job.gpu_id}")
    print("CMD:", " ".join(shlex.quote(x) for x in cmd))

    if not args.dry_run:
        proc = subprocess.run(cmd, env=env, check=False)
        rc = proc.returncode
        status = "OK" if rc == 0 else "FAIL"

    dur = time.time() - start
    
    # 학습 결과 디렉토리 찾기 (수정된 train 스크립트에서 더 직관적인 이름 사용)
    run_dir = None
    out_base = Path(job.output_dir)
    
    # train_으로 시작하는 최신 디렉토리 찾기
    candidates = sorted(out_base.glob("train_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        run_dir = str(candidates[0])

    row = {
        "idx": idx + 1,
        "dataset": job.dataset,
        "model": job.model,
        "epochs": job.epochs,
        "gpu": job.gpu_id,
        "run_name": job.run_name,
        "status": status,
        "returncode": rc,
        "duration_sec": f"{dur:.1f}",
        "run_dir": run_dir,
    }
    
    # sweep 파라미터들 추가
    for k, v in job.sweep_args.items():
        row[k] = v

    append_csv_row(csv_path, header, row)
    return row

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    random.seed(42)

    jobs, sweep_keys = expand_grid(args)
    print(f"Total jobs: {len(jobs)}")

    gpu_cycle = parse_list_arg(args.gpus, str)
    if gpu_cycle:
        print(f"Using GPUs (round-robin): {gpu_cycle}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep_results.csv"
    header = ensure_csv_header(csv_path, sweep_keys)

    if args.dry_run:
        for i, job in enumerate(jobs):
            if gpu_cycle:
                job.gpu_id = gpu_cycle[i % len(gpu_cycle)]
            cmd = build_cmd(args, job)
            print(f"[DRY-RUN {i+1}] GPU={job.gpu_id or '-'} CMD:", " ".join(shlex.quote(x) for x in cmd))
        print("Dry-run complete.")
        return

    max_workers = max(1, int(args.max_concurrent))
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one, i, job, args, gpu_cycle, csv_path, header) for i, job in enumerate(jobs)]
        for fut in as_completed(futures):
            res = fut.result()
            if res["status"] == "OK":
                ok += 1
            else:
                fail += 1
            print(f"[Progress] OK={ok} FAIL={fail} / {len(jobs)}")

    print(f"\nSweep finished. Results CSV: {csv_path}")
    
    # 결과 요약 출력
    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            successful_runs = df[df['status'] == 'OK']
            failed_runs = df[df['status'] == 'FAIL']
            
            print(f"\n=== 실험 결과 요약 ===")
            print(f"성공: {len(successful_runs)}/{len(df)} 실험")
            print(f"실패: {len(failed_runs)}/{len(df)} 실험")
            
            if len(successful_runs) > 0:
                print(f"\n성공한 실험 설정들:")
                for _, row in successful_runs.iterrows():
                    params = []
                    for key in sweep_keys:
                        if key in row and str(row[key]) != "":
                            params.append(f"{key}={row[key]}")
                    params_str = ", ".join(params) if params else "기본설정"
                    print(f"  - {row['run_name']}: {params_str}")
                    
            if len(failed_runs) > 0:
                print(f"\n실패한 실험들:")
                for _, row in failed_runs.iterrows():
                    print(f"  - {row['run_name']} (code: {row['returncode']})")
                    
        except ImportError:
            print("pandas가 설치되지 않아 상세 요약을 표시할 수 없습니다.")
        except Exception as e:
            print(f"결과 요약 생성 중 오류: {e}")

if __name__ == "__main__":
    main()