#!/usr/bin/env python3
"""
🎯 YOLOv8 모델 종합 평가 시스템
학습된 모델의 성능을 다각도로 분석하고 아름다운 시각화 결과를 생성합니다.

주요 기능:
- 정량적 성능 지표 계산 (mAP, Precision, Recall)
- 시각적 예측 결과 분석
- 클래스별 성능 심층 분석
- 전문적인 평가 보고서 생성
"""

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import argparse
import yaml
import logging
import warnings
import koreanize_matplotlib

# 시각화 스타일 설정 - 전문적이고 아름다운 차트를 위한 기본 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# 📊 시각화 상수 정의 - 일관된 디자인을 위한 색상과 스타일
class VisualConfig:
    """시각화 설정을 중앙에서 관리하는 클래스"""
    
    # 색상 팔레트 - 과실 검출에 적합한 자연스러운 색상들
    PRIMARY_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    BACKGROUND_COLOR = '#F8F9FA'
    TEXT_COLOR = '#2C3E50'
    GRID_COLOR = '#E9ECEF'
    
    # 폰트 설정
    TITLE_FONT_SIZE = 16
    LABEL_FONT_SIZE = 12
    TICK_FONT_SIZE = 10
    
    # 차트 크기 설정
    LARGE_FIGURE_SIZE = (16, 10)
    MEDIUM_FIGURE_SIZE = (12, 8)
    SMALL_FIGURE_SIZE = (10, 6)

def setup_logging(output_dir: Path, fruit_name: Optional[str] = None) -> logging.Logger:
    """
    🔧 평가 과정을 추적하기 위한 로깅 시스템 설정
    
    Args:
        output_dir: 로그 파일이 저장될 디렉토리
        fruit_name: 과실명 (로그 파일명에 포함)
    
    Returns:
        설정된 로거 객체
    """
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
    """
    🤖 모델을 안전하게 로드하고 기본 정보를 출력
    
    Args:
        model_path: 모델 파일 경로
        
    Returns:
        로드된 YOLO 모델
        
    Raises:
        FileNotFoundError: 모델 파일이 존재하지 않는 경우
    """
    if not model_path.exists():
        raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
    
    print(f"🔄 모델 로드 중: {model_path.name}")
    model = YOLO(str(model_path))
    
    # 모델 기본 정보 출력 - 사용자가 올바른 모델을 로드했는지 확인
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
    """
    📈 종합적인 성능 지표 계산 - 단순한 숫자를 넘어 의미 있는 인사이트 제공
    
    Args:
        model: 평가할 YOLO 모델
        data_yaml_path: 데이터셋 설정 파일
        conf_threshold: 신뢰도 임계값
        iou_threshold: IoU 임계값
        
    Returns:
        계산된 모든 성능 지표를 담은 딕셔너리
    """
    print("\n🔍 === 성능 지표 계산 시작 ===")
    
    # YOLO 내장 평가 함수 실행 - 가장 정확하고 표준적인 방법
    results = model.val(
        data=str(data_yaml_path),
        split='test',
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,  # 불필요한 출력 억제
        save_json=True   # 상세 결과를 JSON으로 저장
    )
    
    # 핵심 지표 추출 및 정리
    metrics = {
        'overall': {
            'map50': float(results.box.map50),      # IoU 0.5에서의 mAP
            'map50_95': float(results.box.map),     # IoU 0.5-0.95에서의 mAP
            'precision': float(results.box.mp),     # 전체 정밀도
            'recall': float(results.box.mr),        # 전체 재현율
            'f1_score': 2 * float(results.box.mp) * float(results.box.mr) / (float(results.box.mp) + float(results.box.mr)) if (float(results.box.mp) + float(results.box.mr)) > 0 else 0
        },
        'class_wise': {},
        'evaluation_settings': {
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'evaluation_time': datetime.now().isoformat()
        }
    }
    
    # 클래스별 성능 분석 - 어떤 과실을 가장 잘/못 찾는지 파악
    class_names = model.names
    if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
        for idx, class_idx in enumerate(results.box.ap_class_index):
            if idx < len(results.box.ap50):
                class_name = class_names[class_idx]
                metrics['class_wise'][class_name] = {
                    'map50': float(results.box.ap50[idx]),
                    'class_id': int(class_idx)
                }
    
    # 성능 평가 출력 - 사용자가 한눈에 결과를 파악할 수 있도록
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

def create_beautiful_visualizations(model: YOLO, test_images_dir: Path, 
                                  class_names: List[str], metrics: Dict[str, Any],
                                  output_dir: Path, num_samples: int = 20, 
                                  conf_threshold: float = 0.25) -> None:
    """
    🎨 전문적이고 아름다운 시각화 생성 - 논문이나 발표에 사용할 수 있는 품질
    
    Args:
        model: YOLO 모델
        test_images_dir: 테스트 이미지 디렉토리
        class_names: 클래스 이름 리스트
        metrics: 계산된 성능 지표
        output_dir: 결과 저장 디렉토리
        num_samples: 시각화할 샘플 수
        conf_threshold: 신뢰도 임계값
    """
    print(f"\n🎨 === 시각화 생성 시작 (샘플 {num_samples}개) ===")
    
    # 테스트 이미지 파일 수집 - 하위 폴더까지 재귀적으로 검색
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    # 지정된 디렉토리와 하위 폴더에서 이미지 파일 수집
    for ext in image_extensions:
        image_files.extend(list(test_images_dir.rglob(ext)))  # rglob으로 하위 폴더까지 검색
    
    if not image_files:
        print(f"⚠️ 테스트 이미지를 찾을 수 없습니다: {test_images_dir}")
        return
    
    print(f"📁 찾은 이미지 파일: {len(image_files)}개")
    
    # 샘플 이미지 선택 - 다양한 케이스를 보여주기 위해 균등하게 선택
    selected_images = image_files[:num_samples] if len(image_files) >= num_samples else image_files
    
    # 1️⃣ 예측 결과 샘플 시각화 - 실제 검출 결과를 직관적으로 보여줌
    _create_prediction_samples_visualization(model, selected_images, class_names, 
                                           conf_threshold, output_dir)
    
    # 2️⃣ 성능 지표 대시보드 - 한 눈에 모든 성능을 파악할 수 있는 종합 차트
    _create_performance_dashboard(metrics, class_names, output_dir)
    
    # 3️⃣ 신뢰도 분석 차트 - 모델이 얼마나 확신을 가지고 예측하는지 분석
    _create_confidence_analysis(model, selected_images, class_names, 
                               conf_threshold, output_dir)

def _create_prediction_samples_visualization(model: YOLO, image_files: List[Path], 
                                           class_names: List[str], conf_threshold: float,
                                           output_dir: Path) -> None:
    """예측 결과 샘플 시각화 - 실제 검출 결과를 보기 좋게 정리"""
    
    # 격자 레이아웃 계산 - 이미지 개수에 따라 최적의 배치 결정
    n_images = min(len(image_files), 20)  # 최대 20개까지만 표시
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    fig.suptitle('🔍 모델 예측 결과 샘플', fontsize=VisualConfig.TITLE_FONT_SIZE, 
                 fontweight='bold', y=0.98)
    
    # 단일 이미지인 경우 axes를 리스트로 변환
    if n_images == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, image_path in enumerate(image_files[:n_images]):
        try:
            # 이미지 로드 및 예측
            image = cv2.imread(str(image_path))
            if image is None:
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = model(image_path, conf=conf_threshold, verbose=False)
            
            # 현재 subplot 설정
            ax = axes[idx] if isinstance(axes, (list, np.ndarray)) else axes
            ax.imshow(image_rgb)
            ax.set_title(f"{image_path.stem}", fontsize=VisualConfig.LABEL_FONT_SIZE)
            ax.axis('off')
            
            # 예측 결과 시각화
            detection_count = 0
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    class_id = int(box.cls)
                    
                    if conf >= conf_threshold:
                        # 클래스별로 다른 색상 사용
                        color = VisualConfig.PRIMARY_COLORS[class_id % len(VisualConfig.PRIMARY_COLORS)]
                        
                        # 바운딩 박스
                        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                           linewidth=3, edgecolor=color,
                                           facecolor='none', alpha=0.8)
                        ax.add_patch(rect)
                        
                        # 레이블 - 가독성을 위한 배경 추가
                        label = f"{class_names[class_id]} {conf:.2f}"
                        ax.text(x1, y1-10, label, 
                               color='white', fontweight='bold', fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.3", 
                                       facecolor=color, alpha=0.8))
                        detection_count += 1
            
            # 검출 개수 표시
            ax.text(0.02, 0.98, f"검출: {detection_count}개", 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor='white', alpha=0.7))
            
        except Exception as e:
            print(f"⚠️ 이미지 처리 오류: {image_path.name} - {e}")
            continue
    
    # 빈 subplot 숨기기
    for idx in range(n_images, len(axes)):
        if isinstance(axes, (list, np.ndarray)) and idx < len(axes):
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_samples.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ 예측 샘플 저장: prediction_samples.png")

def _create_performance_dashboard(metrics: Dict[str, Any], class_names: List[str], 
                                 output_dir: Path) -> None:
    """성능 지표 종합 대시보드 생성 - 전문적인 분석 차트"""
    
    fig = plt.figure(figsize=VisualConfig.LARGE_FIGURE_SIZE)
    fig.suptitle('📊 모델 성능 종합 대시보드', fontsize=18, fontweight='bold')
    
    # 1. 전체 성능 지표 바 차트
    ax1 = plt.subplot(2, 3, 1)
    overall_metrics = metrics['overall']
    metric_names = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1-Score']
    metric_values = [overall_metrics['map50'], overall_metrics['map50_95'], 
                    overall_metrics['precision'], overall_metrics['recall'], 
                    overall_metrics['f1_score']]
    
    bars = ax1.bar(metric_names, metric_values, color=VisualConfig.PRIMARY_COLORS[:len(metric_names)],
                   alpha=0.8, edgecolor='white', linewidth=2)
    ax1.set_title('전체 성능 지표', fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # 값 레이블 추가
    for bar, value in zip(bars, metric_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    
    # 2. 클래스별 성능 비교 (mAP50)
    if metrics['class_wise']:
        ax2 = plt.subplot(2, 3, 2)
        class_names_list = list(metrics['class_wise'].keys())
        class_scores = [metrics['class_wise'][name]['map50'] for name in class_names_list]
        
        bars = ax2.barh(class_names_list, class_scores, 
                       color=VisualConfig.PRIMARY_COLORS[:len(class_names_list)])
        ax2.set_title('클래스별 mAP50', fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.grid(axis='x', alpha=0.3)
        
        # 값 레이블 추가
        for bar, score in zip(bars, class_scores):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center', fontweight='bold')
    
    # 3. 성능 등급 도넛 차트
    ax3 = plt.subplot(2, 3, 3)
    map50 = overall_metrics['map50']
    
    if map50 >= 0.8:
        grade, color = '우수', '#2ECC71'
    elif map50 >= 0.6:
        grade, color = '양호', '#F39C12'
    elif map50 >= 0.4:
        grade, color = '보통', '#E67E22'
    else:
        grade, color = '개선필요', '#E74C3C'
    
    # 도넛 차트 생성
    wedges, texts = ax3.pie([map50, 1-map50], colors=[color, '#ECF0F1'], 
                           startangle=90, counterclock=False,
                           wedgeprops=dict(width=0.5))
    
    ax3.text(0, 0, f'{grade}\n{map50:.3f}', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    ax3.set_title('성능 등급', fontweight='bold')
    
    # 4-6. 추가 분석 차트들 (공간 활용)
    ax4 = plt.subplot(2, 3, (4, 6))
    
    # 성능 해석 가이드 텍스트
    interpretation_text = f"""
🎯 성능 해석 가이드

mAP50: {overall_metrics['map50']:.3f} ({grade})
• 0.8 이상: 상용화 가능한 우수한 성능
• 0.6-0.8: 실용적 활용 가능한 양호한 성능  
• 0.4-0.6: 개선을 통해 활용 가능한 보통 성능
• 0.4 미만: 추가 학습이 필요한 성능

균형 분석:
• 정밀도 vs 재현율: {overall_metrics['precision']:.3f} vs {overall_metrics['recall']:.3f}
• F1-Score: {overall_metrics['f1_score']:.3f}

💡 개선 방향:
• 정밀도가 낮으면: False Positive 줄이기 (임계값 조정)
• 재현율이 낮으면: False Negative 줄이기 (더 많은 데이터)
• 둘 다 낮으면: 모델 용량 확대 또는 학습 시간 증가
"""
    
    ax4.text(0.05, 0.95, interpretation_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', alpha=0.8))
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_dashboard.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ 성능 대시보드 저장: performance_dashboard.png")

def _create_confidence_analysis(model: YOLO, image_files: List[Path], 
                               class_names: List[str], conf_threshold: float,
                               output_dir: Path) -> None:
    """신뢰도 분석 차트 생성 - 모델의 확신 정도 분석"""
    
    print("🔍 신뢰도 분석 중...")
    
    # 모든 예측에서 신뢰도 정보 수집
    all_confidences = []
    class_confidences = defaultdict(list)
    detection_counts = []
    
    for image_path in image_files[:50]:  # 신뢰도 분석은 50개 이미지로 제한
        try:
            results = model(image_path, conf=0.01, verbose=False)  # 낮은 임계값으로 모든 예측 수집
            
            image_detections = 0
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for box in boxes:
                    conf = float(box.conf)
                    class_id = int(box.cls)
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                    
                    all_confidences.append(conf)
                    class_confidences[class_name].append(conf)
                    
                    if conf >= conf_threshold:
                        image_detections += 1
            
            detection_counts.append(image_detections)
            
        except Exception as e:
            continue
    
    if not all_confidences:
        print("⚠️ 신뢰도 분석을 위한 데이터가 부족합니다.")
        return
    
    # 신뢰도 분석 시각화
    fig, axes = plt.subplots(2, 2, figsize=VisualConfig.LARGE_FIGURE_SIZE)
    fig.suptitle('🎪 모델 신뢰도 분석', fontsize=16, fontweight='bold')
    
    # 1. 전체 신뢰도 분포 히스토그램
    ax1 = axes[0, 0]
    ax1.hist(all_confidences, bins=30, alpha=0.7, color=VisualConfig.PRIMARY_COLORS[0],
            edgecolor='white', linewidth=1)
    ax1.axvline(conf_threshold, color='red', linestyle='--', linewidth=2,
               label=f'임계값: {conf_threshold}')
    ax1.set_xlabel('신뢰도')
    ax1.set_ylabel('빈도')
    ax1.set_title('신뢰도 분포')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 통계 정보 추가
    mean_conf = np.mean(all_confidences)
    ax1.text(0.7, 0.8, f'평균: {mean_conf:.3f}\n표준편차: {np.std(all_confidences):.3f}',
            transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3", 
                                              facecolor='white', alpha=0.8))
    
    # 2. 클래스별 신뢰도 박스플롯
    ax2 = axes[0, 1]
    if class_confidences:
        class_data = []
        class_labels = []
        for class_name, confs in class_confidences.items():
            if len(confs) >= 5:  # 충분한 데이터가 있는 클래스만
                class_data.append(confs)
                class_labels.append(class_name)
        
        if class_data:
            bp = ax2.boxplot(class_data, labels=class_labels, patch_artist=True)
            
            # 박스플롯 색상 설정
            for patch, color in zip(bp['boxes'], VisualConfig.PRIMARY_COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax2.set_ylabel('신뢰도')
            ax2.set_title('클래스별 신뢰도 분포')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(alpha=0.3)
    
    # 3. 이미지당 검출 개수 분포
    ax3 = axes[1, 0]
    detection_counter = Counter(detection_counts)
    counts = sorted(detection_counter.keys())
    frequencies = [detection_counter[c] for c in counts]
    
    bars = ax3.bar(counts, frequencies, color=VisualConfig.PRIMARY_COLORS[2], alpha=0.7,
                  edgecolor='white', linewidth=1)
    ax3.set_xlabel('이미지당 검출 개수')
    ax3.set_ylabel('이미지 수')
    ax3.set_title('검출 개수 분포')
    ax3.grid(alpha=0.3)
    
    # 평균 검출 개수 표시
    mean_detections = np.mean(detection_counts)
    ax3.axvline(mean_detections, color='red', linestyle='--', linewidth=2,
               label=f'평균: {mean_detections:.1f}개')
    ax3.legend()
    
    # 4. 신뢰도 임계값별 성능 시뮬레이션
    ax4 = axes[1, 1]
    thresholds = np.arange(0.1, 1.0, 0.05)
    detection_rates = []
    
    for threshold in thresholds:
        valid_detections = sum(1 for conf in all_confidences if conf >= threshold)
        detection_rate = valid_detections / len(all_confidences) if all_confidences else 0
        detection_rates.append(detection_rate)
    
    ax4.plot(thresholds, detection_rates, linewidth=3, color=VisualConfig.PRIMARY_COLORS[3],
            marker='o', markersize=4)
    ax4.axvline(conf_threshold, color='red', linestyle='--', linewidth=2,
               label=f'현재 임계값: {conf_threshold}')
    ax4.set_xlabel('신뢰도 임계값')
    ax4.set_ylabel('검출 비율')
    ax4.set_title('임계값별 검출 비율')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ 신뢰도 분석 저장: confidence_analysis.png")

def generate_professional_report(model_path: Path, metrics: Dict[str, Any], 
                               data_config: Dict[str, Any], output_dir: Path,
                               fruit_name: Optional[str] = None) -> None:
    """
    📋 전문적인 평가 보고서 생성 - 연구나 비즈니스 목적으로 사용 가능한 수준
    
    Args:
        model_path: 평가된 모델 경로
        metrics: 계산된 성능 지표
        data_config: 데이터셋 설정 정보
        output_dir: 보고서 저장 디렉토리
        fruit_name: 과실명 (보고서 제목에 포함)
    """
    
    # 보고서 메타데이터
    eval_time = datetime.now()
    fruit_title = f"{fruit_name} " if fruit_name else ""
    
    # 성능 등급 계산
    map50 = metrics['overall']['map50']
    if map50 >= 0.8:
        performance_grade = "우수 (Excellent)"
        recommendation = "상용화 준비 완료"
    elif map50 >= 0.6:
        performance_grade = "양호 (Good)"
        recommendation = "실용적 활용 가능"
    elif map50 >= 0.4:
        performance_grade = "보통 (Fair)"
        recommendation = "추가 개선 권장"
    else:
        performance_grade = "개선필요 (Needs Improvement)"
        recommendation = "상당한 개선 필요"
    
    # 클래스별 분석
    class_analysis = ""
    if metrics['class_wise']:
        sorted_classes = sorted(metrics['class_wise'].items(), 
                              key=lambda x: x[1]['map50'], reverse=True)
        
        best_class = sorted_classes[0]
        worst_class = sorted_classes[-1]
        
        class_analysis = f"""
### 클래스별 성능 분석

**최고 성능 클래스**: {best_class[0]} (mAP50: {best_class[1]['map50']:.3f})
**개선 필요 클래스**: {worst_class[0]} (mAP50: {worst_class[1]['map50']:.3f})

| 클래스 | mAP50 | 성능 등급 |
|--------|-------|-----------|"""
        
        for class_name, class_metrics in sorted_classes:
            class_map = class_metrics['map50']
            if class_map >= 0.7:
                class_grade = "우수"
            elif class_map >= 0.5:
                class_grade = "양호"
            elif class_map >= 0.3:
                class_grade = "보통"
            else:
                class_grade = "개선필요"
            
            class_analysis += f"\n| {class_name} | {class_map:.3f} | {class_grade} |"
    
    # 상세 보고서 내용
    report_content = f"""# 🎯 {fruit_title}YOLOv8 객체 검출 모델 성능 평가 보고서

**평가 일시**: {eval_time.strftime('%Y년 %m월 %d일 %H:%M:%S')}  
**모델 파일**: `{model_path.name}`  
**데이터셋**: `{data_config.get('path', 'Unknown')}`  
**평가자**: 자동 평가 시스템  

---

## 📊 성능 요약

### 전체 성능 지표
- **mAP50**: {metrics['overall']['map50']:.4f}
- **mAP50-95**: {metrics['overall']['map50_95']:.4f}
- **정밀도 (Precision)**: {metrics['overall']['precision']:.4f}
- **재현율 (Recall)**: {metrics['overall']['recall']:.4f}
- **F1-Score**: {metrics['overall']['f1_score']:.4f}

### 종합 평가
**성능 등급**: {performance_grade}  
**권장사항**: {recommendation}

---

## 🔍 상세 분석

### 성능 지표 해석

**mAP50 ({metrics['overall']['map50']:.3f})**
- 이 값은 IoU 임계값 0.5에서의 평균 정밀도입니다
- 값이 높을수록 모델이 객체를 정확하게 찾고 분류하는 능력이 우수합니다
- 0.8 이상이면 상용화 가능한 우수한 성능입니다

**정밀도 vs 재현율 균형**
- 정밀도 {metrics['overall']['precision']:.3f}: 모델이 예측한 것 중 실제로 맞는 비율
- 재현율 {metrics['overall']['recall']:.3f}: 실제 객체 중 모델이 찾아낸 비율
- F1-Score {metrics['overall']['f1_score']:.3f}: 정밀도와 재현율의 조화평균

{class_analysis}

---

## 🎯 개선 방안

### 단기 개선 (1-2주)
1. **하이퍼파라미터 튜닝**
   - 신뢰도 임계값 조정: 현재 {metrics['evaluation_settings']['conf_threshold']}
   - IoU 임계값 최적화: 현재 {metrics['evaluation_settings']['iou_threshold']}
   - 학습률 스케줄 조정

2. **데이터 증강 강화**
   - 다양한 조명 조건 추가
   - 회전, 크기 변경, 색상 조정 적용
   - 배경 다양화

### 중기 개선 (1-2개월)
1. **데이터셋 확장**
   - 부족한 클래스의 데이터 추가 수집
   - 어려운 케이스 (겹침, 부분 가림) 데이터 보강
   - 다양한 환경에서의 데이터 수집

2. **모델 구조 개선**
   - 더 큰 YOLOv8 모델 (s → m → l → x) 실험
   - 앙상블 기법 적용
   - 커스텀 백본 네트워크 검토

### 장기 개선 (3-6개월)
1. **도메인 특화 최적화**
   - 농업/과실 특화 전처리 기법 개발
   - 시간대별, 계절별 모델 개발
   - 실시간 처리 최적화

2. **배포 환경 최적화**
   - 모바일/엣지 디바이스 최적화
   - 클라우드 API 개발
   - 실시간 모니터링 시스템 구축

---

## 📈 성능 기준 및 벤치마크

| 성능 등급 | mAP50 범위 | 활용 가능성 | 권장 조치 |
|-----------|------------|-------------|-----------|
| 우수 | 0.8 이상 | 즉시 상용화 가능 | 유지보수 및 모니터링 |
| 양호 | 0.6 - 0.8 | 실용적 활용 가능 | 소폭 개선 후 활용 |
| 보통 | 0.4 - 0.6 | 제한적 활용 가능 | 상당한 개선 필요 |
| 개선필요 | 0.4 미만 | 추가 개발 필요 | 전면적 재검토 |

---

## 🔧 기술적 권장사항

### 즉시 적용 가능한 개선
- 신뢰도 임계값을 {max(0.1, metrics['evaluation_settings']['conf_threshold'] - 0.05):.2f}로 낮춰서 재현율 향상 시도
- 테스트 시간 증강(TTA) 적용으로 성능 향상
- 후처리 알고리즘 최적화

### 데이터 관련 권장사항
- 클래스 불균형 해결을 위한 가중치 조정
- 하드 네거티브 마이닝 적용
- 크로스 검증을 통한 안정성 확인

---

## 📊 부록: 상세 통계

### 평가 환경
- **평가 일시**: {eval_time.isoformat()}
- **신뢰도 임계값**: {metrics['evaluation_settings']['conf_threshold']}
- **IoU 임계값**: {metrics['evaluation_settings']['iou_threshold']}
- **클래스 수**: {len(data_config.get('names', []))}

### 성능 메트릭 상세
```
전체 성능:
  mAP50    : {metrics['overall']['map50']:.6f}
  mAP50-95 : {metrics['overall']['map50_95']:.6f}
  Precision: {metrics['overall']['precision']:.6f}
  Recall   : {metrics['overall']['recall']:.6f}
  F1-Score : {metrics['overall']['f1_score']:.6f}
```

---

**보고서 생성**: 자동 평가 시스템 v2.0  
**문의**: 모델 개발팀  
**다음 평가 예정**: 모델 개선 후 재평가 권장
"""

    # 보고서 파일 저장
    fruit_suffix = f"_{fruit_name}" if fruit_name else ""
    report_filename = f"evaluation_report{fruit_suffix}_{eval_time.strftime('%Y%m%d_%H%M%S')}.md"
    report_path = output_dir / report_filename
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # CSV 형태의 수치 데이터도 저장
    metrics_df = pd.DataFrame([
        {
            'metric': 'mAP50',
            'value': metrics['overall']['map50'],
            'category': 'overall'
        },
        {
            'metric': 'mAP50-95', 
            'value': metrics['overall']['map50_95'],
            'category': 'overall'
        },
        {
            'metric': 'Precision',
            'value': metrics['overall']['precision'], 
            'category': 'overall'
        },
        {
            'metric': 'Recall',
            'value': metrics['overall']['recall'],
            'category': 'overall'
        },
        {
            'metric': 'F1-Score',
            'value': metrics['overall']['f1_score'],
            'category': 'overall'
        }
    ])
    
    # 클래스별 데이터 추가
    for class_name, class_metrics in metrics.get('class_wise', {}).items():
        metrics_df = pd.concat([metrics_df, pd.DataFrame([{
            'metric': 'mAP50',
            'value': class_metrics['map50'],
            'category': class_name
        }])], ignore_index=True)
    
    csv_filename = f"evaluation_metrics{fruit_suffix}_{eval_time.strftime('%Y%m%d_%H%M%S')}.csv"
    metrics_df.to_csv(output_dir / csv_filename, index=False, encoding='utf-8')
    
    print(f"📋 평가 보고서 저장: {report_filename}")
    print(f"📊 수치 데이터 저장: {csv_filename}")

def main():
    """🚀 메인 실행 함수 - 전체 평가 파이프라인 조율"""
    
    parser = argparse.ArgumentParser(
        description='🎯 YOLOv8 모델 종합 성능 평가 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python evaluate_model.py --model best.pt --data data.yaml
  python evaluate_model.py --model best_apple.pt --data data.yaml --fruit apple --samples 50
  python evaluate_model.py --model model.pt --data data.yaml --conf_threshold 0.3 --output_dir results
  python evaluate_model.py --model best.pt --data data.yaml --test_images_dir /path/to/test/images
        """
    )
    
    # 필수 인자
    parser.add_argument('--model', required=True, help='평가할 모델 파일 경로 (.pt)')
    parser.add_argument('--data', required=True, help='데이터셋 설정 파일 (data.yaml)')
    
    # 선택적 인자
    parser.add_argument('--output_dir', default='evaluation_results', 
                       help='평가 결과 저장 경로 (기본값: evaluation_results)')
    parser.add_argument('--test_images_dir', type=str,
                       help='테스트 이미지 디렉토리 경로 (지정하지 않으면 data.yaml의 경로 사용)')
    parser.add_argument('--fruit', type=str, 
                       help='과실명 (결과 파일명에 포함, 선택사항)')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='신뢰도 임계값 (기본값: 0.25)')
    parser.add_argument('--iou_threshold', type=float, default=0.45,
                       help='IoU 임계값 (기본값: 0.45)')
    parser.add_argument('--samples', type=int, default=20,
                       help='시각화할 샘플 이미지 수 (기본값: 20)')
    
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
        print(f"\n🎯 === YOLOv8 모델 종합 평가 시작 ===")
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
        
        # 4️⃣ 테스트 이미지 디렉토리 설정
        if args.test_images_dir:
            # 사용자가 지정한 경로 사용
            test_images_dir = Path(args.test_images_dir)
            print(f"🖼️ 사용자 지정 테스트 이미지 경로: {test_images_dir}")
        else:
            # 기본 경로 사용 (data.yaml 기준)
            test_images_dir = Path(data_config['path']) / 'images' / 'test'
            print(f"🖼️ 기본 테스트 이미지 경로: {test_images_dir}")
        
        # 5️⃣ 시각화 생성
        if test_images_dir.exists():
            create_beautiful_visualizations(
                model, test_images_dir, class_names, metrics, 
                output_dir, args.samples, args.conf_threshold
            )
        else:
            logger.warning(f"⚠️ 테스트 이미지 디렉토리를 찾을 수 없습니다: {test_images_dir}")
            print(f"⚠️ 다음 경로들을 확인해주세요:")
            print(f"   - 지정된 경로: {test_images_dir}")
            if args.test_images_dir:
                default_path = Path(data_config['path']) / 'images' / 'test'
                print(f"   - 기본 경로: {default_path}")
        
        # 6️⃣ 전문 보고서 생성
        generate_professional_report(
            model_path, metrics, data_config, output_dir, args.fruit
        )
        
        # 최종 결과 요약
        print(f"\n🎉 === 평가 완료 ===")
        print(f"📊 전체 성능 (mAP50): {metrics['overall']['map50']:.3f}")
        print(f"📁 모든 결과가 저장되었습니다: {output_dir}")
        print(f"📋 상세 보고서를 확인하세요!")
        
        logger.info(f"평가 완료 - mAP50: {metrics['overall']['map50']:.3f}")
        
    except Exception as e:
        error_msg = f"💥 평가 중 오류 발생: {e}"
        print(error_msg)
        logger.error(error_msg)
        raise

if __name__ == "__main__":
    main()