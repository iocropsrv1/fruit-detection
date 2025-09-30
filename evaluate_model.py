#!/usr/bin/env python3
"""
🎯 YOLOv8 모델 종합 평가 시스템 (CSV 출력 최적화)
학습된 모델의 성능을 평가하고 CSV 결과만 생성합니다.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse
import yaml
import logging

def setup_logging(output_dir: Path, fruit_name: Optional[str] = None) -> logging.Logger:
    """평가 과정을 추적하기 위한 로깅 시스템 설정"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fruit_suffix = f"_{fruit_name}" if fruit_name else ""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_dir / f"evaluation_{timestamp}{fruit_suffix}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"🎯 모델 평가 시작 - {fruit_name or '일반'} 모드")
    return logger

def load_model_safely(model_path: Path) -> YOLO:
    """모델을 안전하게 로드하고 기본 정보를 출력"""
    if not model_path.exists():
        raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
    
    print(f"🔄 모델 로드 중: {model_path.name}")
    model = YOLO(str(model_path))
    
    # 모델 기본 정보 출력
    try:
        model_info = model.info(verbose=False)
        print(f"✅ 모델 로드 완료")
        print(f"   📏 모델 크기: {model_path.stat().st_size / (1024*1024):.1f}MB")
        print(f"   🏷️  클래스 수: {len(model.names)}개")
        print(f"   📝 클래스: {', '.join(list(model.names.values())[:5])}{'...' if len(model.names) > 5 else ''}")
    except:
        print(f"⚠️ 모델 정보를 가져올 수 없지만 로드는 성공했습니다.")
    
    return model

def calculate_comprehensive_metrics(model: YOLO, data_yaml_path: Path, 
                                  conf_threshold: float = 0.25, 
                                  iou_threshold: float = 0.45) -> Dict[str, Any]:
    """종합적인 성능 지표 계산"""
    print("\n🔍 === 성능 지표 계산 시작 ===")
    
    # YOLO 내장 평가 함수 실행
    results = model.val(
        data=str(data_yaml_path),
        split='test',
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
        save_json=True
    )
    
    # 핵심 지표 추출 및 정리
    metrics = {
        'overall': {
            'map50': float(results.box.map50),
            'map50_95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1_score': 2 * float(results.box.mp) * float(results.box.mr) / (float(results.box.mp) + float(results.box.mr)) if (float(results.box.mp) + float(results.box.mr)) > 0 else 0
        },
        'class_wise': {},
        'evaluation_settings': {
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'evaluation_time': datetime.now().isoformat()
        }
    }
    
    # 클래스별 성능 분석
    class_names = model.names
    if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
        for idx, class_idx in enumerate(results.box.ap_class_index):
            if idx < len(results.box.ap50):
                class_name = class_names[class_idx]
                metrics['class_wise'][class_name] = {
                    'map50': float(results.box.ap50[idx]),
                    'class_id': int(class_idx)
                }
    
    # 성능 평가 출력
    print(f"📊 전체 성능 요약:")
    print(f"   🎯 mAP50: {metrics['overall']['map50']:.3f} ({'우수' if metrics['overall']['map50'] > 0.7 else '양호' if metrics['overall']['map50'] > 0.5 else '개선필요'})")
    print(f"   📏 mAP50-95: {metrics['overall']['map50_95']:.3f}")
    print(f"   🎪 정밀도: {metrics['overall']['precision']:.3f}")
    print(f"   🔍 재현율: {metrics['overall']['recall']:.3f}")
    print(f"   ⚖️ F1-Score: {metrics['overall']['f1_score']:.3f}")
    
    if metrics['class_wise']:
        print(f"\n🏷️ 클래스별 성능 (mAP50):")
        sorted_classes = sorted(metrics['class_wise'].items(), 
                              key=lambda x: x[1]['map50'], reverse=True)
        for class_name, class_metrics in sorted_classes:
            performance_emoji = "🏆" if class_metrics['map50'] > 0.8 else "🥈" if class_metrics['map50'] > 0.6 else "🥉" if class_metrics['map50'] > 0.4 else "📈"
            print(f"   {performance_emoji} {class_name}: {class_metrics['map50']:.3f}")
    
    return metrics

def generate_csv_results(model_path: Path, metrics: Dict[str, Any], 
                        data_config: Dict[str, Any], output_dir: Path,
                        fruit_name: Optional[str] = None) -> None:
    """CSV 결과 파일 생성"""
    
    eval_time = datetime.now()
    fruit_suffix = f"_{fruit_name}" if fruit_name else ""
    
    # 전체 성능 데이터 준비
    results_data = []
    
    # 전체 성능 지표
    overall_metrics = metrics['overall']
    base_data = {
        'model_path': str(model_path),
        'model_name': model_path.name,
        'fruit_name': fruit_name or '',
        'dataset_path': data_config.get('path', 'Unknown'),
        'num_classes': len(data_config.get('names', [])),
        'class_names': ','.join(data_config.get('names', [])),
        'evaluation_time': eval_time.isoformat(),
        'conf_threshold': metrics['evaluation_settings']['conf_threshold'],
        'iou_threshold': metrics['evaluation_settings']['iou_threshold'],
        # 전체 성능 지표
        'map50': overall_metrics['map50'],
        'map50_95': overall_metrics['map50_95'],
        'precision': overall_metrics['precision'],
        'recall': overall_metrics['recall'],
        'f1_score': overall_metrics['f1_score'],
    }
    
    # 클래스별 mAP50 추가 (컬럼으로)
    class_wise = metrics.get('class_wise', {})
    for class_name, class_data in class_wise.items():
        base_data[f'map50_{class_name.lower()}'] = class_data['map50']
    
    # 빠진 클래스들은 빈 값으로 채우기
    all_class_names = data_config.get('names', [])
    for class_name in all_class_names:
        col_name = f'map50_{class_name.lower()}'
        if col_name not in base_data:
            base_data[col_name] = ''
    
    results_data.append(base_data)
    
    # DataFrame 생성
    df = pd.DataFrame(results_data)
    
    # CSV 저장
    csv_filename = f"evaluation_results{fruit_suffix}_{eval_time.strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = output_dir / csv_filename
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"📊 평가 결과 CSV 저장: {csv_filename}")
    
    # 결과 요약 출력
    print(f"\n=== 평가 완료 요약 ===")
    print(f"모델: {model_path.name}")
    print(f"전체 mAP50: {overall_metrics['map50']:.4f}")
    if class_wise:
        print("클래스별 성능:")
        for class_name, class_data in sorted(class_wise.items(), key=lambda x: x[1]['map50'], reverse=True):
            print(f"  - {class_name}: {class_data['map50']:.4f}")

def main():
    """메인 실행 함수"""
    
    parser = argparse.ArgumentParser(
        description='🎯 YOLOv8 모델 성능 평가 시스템 (CSV 출력)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python evaluate_model.py --model best.pt --data data.yaml
  python evaluate_model.py --model best_apple.pt --data data.yaml --fruit apple
  python evaluate_model.py --model model.pt --data data.yaml --conf_threshold 0.3
        """
    )
    
    # 필수 인자
    parser.add_argument('--model', required=True, help='평가할 모델 파일 경로 (.pt)')
    parser.add_argument('--data', required=True, help='데이터셋 설정 파일 (data.yaml)')
    
    # 선택적 인자
    parser.add_argument('--output_dir', default='evaluation_results', 
                       help='평가 결과 저장 경로 (기본값: evaluation_results)')
    parser.add_argument('--fruit', type=str, 
                       help='과실명 (결과 파일명에 포함, 선택사항)')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='신뢰도 임계값 (기본값: 0.25)')
    parser.add_argument('--iou_threshold', type=float, default=0.45,
                       help='IoU 임계값 (기본값: 0.45)')
    
    args = parser.parse_args()
    
    # 경로 설정
    model_path = Path(args.model)
    data_yaml_path = Path(args.data)
    output_dir = Path(args.output_dir)
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 로깅 시스템 초기화
    logger = setup_logging(output_dir, args.fruit)
    
    try:
        print(f"\n🎯 === YOLOv8 모델 평가 시작 ===")
        print(f"📁 모델: {model_path}")
        print(f"📊 데이터: {data_yaml_path}")
        print(f"💾 결과: {output_dir}")
        if args.fruit:
            print(f"🍎 과실: {args.fruit}")
        
        # 1️⃣ 모델 로드
        model = load_model_safely(model_path)
        
        # 2️⃣ 데이터 설정 로드
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"❌ 데이터 설정 파일을 찾을 수 없습니다: {data_yaml_path}")
        
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        class_names = data_config.get('names', [])
        if not class_names:
            raise ValueError("❌ 데이터 설정에서 클래스 이름을 찾을 수 없습니다")
        
        # 3️⃣ 성능 지표 계산
        metrics = calculate_comprehensive_metrics(
            model, data_yaml_path, args.conf_threshold, args.iou_threshold
        )
        
        # 4️⃣ CSV 결과 생성
        generate_csv_results(
            model_path, metrics, data_config, output_dir, args.fruit
        )
        
        # 최종 결과 요약
        print(f"\n🎉 === 평가 완료 ===")
        print(f"📊 전체 성능 (mAP50): {metrics['overall']['map50']:.3f}")
        print(f"📁 결과가 저장되었습니다: {output_dir}")
        
        logger.info(f"평가 완료 - mAP50: {metrics['overall']['map50']:.3f}")
        
    except Exception as e:
        error_msg = f"💥 평가 중 오류 발생: {e}"
        print(error_msg)
        logger.error(error_msg)
        raise

if __name__ == "__main__":
    main()