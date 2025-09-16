#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset1(pepper/train) vs Dataset2(pepper/test) 비교 EDA
- 랜덤 샘플링 기반 이미지/객체/배경 복잡도 분석
- 시각화 및 JSON/TXT 리포트 저장
- numpy 타입 JSON 직렬화 안전화 (np.bool_ 포함)

기본 경로:
  D1 imgs: /data/ioCrops/pepper/dataset/train_v1.1_fruit/images/train
  D1 lbls: /data/ioCrops/pepper/dataset/train_v1.1_fruit/labels/train
  D2 imgs: /home/cat123/yolov8-fruit_detection/yolo_dataset_new/pepper/images/test
  D2 lbls: /home/cat123/yolov8-fruit_detection/yolo_dataset_new/pepper/labels/test
출력:
  /home/cat123/yolov8-fruit_detection/analysis_results
"""

import os
import cv2
import json
import argparse
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless 환경 안정화
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
from skimage import measure, filters
from skimage.feature import local_binary_pattern

import torch
import torchvision.transforms as transforms
from PIL import Image

warnings.filterwarnings('ignore')

# GPU 설정(현재 코드는 torch 텐서를 직접 쓰진 않지만, 디바이스 정보 표기)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class DatasetAnalyzer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 분석 결과 저장용 딕셔너리
        self.analysis_results: Dict[str, Dict[str, Any]] = {
            'dataset1': {},
            'dataset2': {},
            'comparison': {}
        }

    # ----------------------------
    # 데이터 로드
    # ----------------------------
    def load_random_samples(self, image_dir: str, label_dir: str, n_samples: int = 200):
        """랜덤하게 n_samples 개의 이미지와 라벨을 로드"""
        image_files = list(Path(image_dir).glob('*.png'))

        if len(image_files) == 0:
            print(f"Warning: no images found under {image_dir}")
            return [], [], []

        if len(image_files) < n_samples:
            print(f"Warning: Only {len(image_files)} images found, using all available")
            n_samples = len(image_files)

        selected_files = random.sample(image_files, n_samples)

        images: List[np.ndarray] = []
        labels: List[list] = []
        image_paths: List[str] = []

        for img_path in selected_files:
            image = cv2.imread(str(img_path))
            if image is None:
                # 이미지 읽기 실패 시 스킵
                continue
            images.append(image)
            image_paths.append(str(img_path))

            label_path = Path(label_dir) / (img_path.stem + '.txt')
            label_data = []

            if label_path.exists():
                try:
                    with open(label_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, x_center, y_center, width, height = map(float, parts)
                            label_data.append({
                                'class_id': int(class_id),
                                'x_center': float(x_center),
                                'y_center': float(y_center),
                                'width': float(width),
                                'height': float(height)
                            })
                except Exception:
                    # 라벨 파싱 실패 시 빈 리스트로 둠
                    pass
            labels.append(label_data)

        print(f"Loaded {len(images)} images and labels from {image_dir}")
        return images, labels, image_paths

    # ----------------------------
    # 이미지 특성 분석
    # ----------------------------
    def analyze_image_properties(self, images: List[np.ndarray], dataset_name: str):
        results = {
            'brightness': [],
            'contrast': [],
            'saturation': [],
            'hue_distribution': [],
            'resolution': [],
            'aspect_ratio': [],
            'blur_metric': [],
            'noise_level': []
        }

        print(f"Analyzing image properties for {dataset_name}...")

        for i, image in enumerate(images):
            if i % 50 == 0:
                print(f"  Processing image {i+1}/{len(images)}")

            h, w = image.shape[:2]
            results['resolution'].append(float(h * w))
            results['aspect_ratio'].append(float(w / max(h, 1)))

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 밝기/채도
            brightness = float(np.mean(hsv[:, :, 2]))
            results['brightness'].append(brightness)

            saturation = float(np.mean(hsv[:, :, 1]))
            results['saturation'].append(saturation)

            # 색조 분포
            hue_hist = np.histogram(hsv[:, :, 0], bins=36, range=(0, 180))[0]
            results['hue_distribution'].append(hue_hist.astype(np.int64))

            # 대비/블러 (Laplacian 분산)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            results['contrast'].append(lap_var)
            results['blur_metric'].append(lap_var)

            # 노이즈 레벨 (고주파 대략)
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1.0)
            noise_level = float(np.std(magnitude_spectrum))
            results['noise_level'].append(noise_level)

        # 통계 계산
        stats_results: Dict[str, Any] = {}
        for key, values in results.items():
            if key != 'hue_distribution' and len(values) > 0:
                arr = np.asarray(values, dtype=np.float64)
                stats_results[key] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'median': float(np.median(arr))
                }

        # 색조 분포 평균
        if len(results['hue_distribution']) > 0:
            mean_hist = np.mean(np.stack(results['hue_distribution'], axis=0), axis=0)
            stats_results['hue_distribution'] = {
                'mean_histogram': mean_hist.astype(np.float64).tolist()
            }
        else:
            stats_results['hue_distribution'] = {'mean_histogram': []}

        self.analysis_results[dataset_name]['image_properties'] = stats_results
        return results

    # ----------------------------
    # 객체 특성 분석
    # ----------------------------
    def analyze_object_properties(self, labels: List[list], images: List[np.ndarray], dataset_name: str):
        results = {
            'class_distribution': [],
            'object_sizes': [],
            'object_positions': [],
            'objects_per_image': [],
            'aspect_ratios': [],
            'center_positions': []
        }

        print(f"Analyzing object properties for {dataset_name}...")

        for label_list, image in zip(labels, images):
            h, w = image.shape[:2]

            # 이미지당 객체 수
            results['objects_per_image'].append(int(len(label_list)))

            for label in label_list:
                # 클래스
                results['class_distribution'].append(int(label['class_id']))

                # 절대 크기
                abs_w = float(label['width'] * w)
                abs_h = float(label['height'] * h)
                area = float(abs_w * abs_h)
                results['object_sizes'].append(area)
                results['aspect_ratios'].append(float(abs_w / max(abs_h, 1e-6)))

                # 위치
                abs_x = float(label['x_center'] * w)
                abs_y = float(label['y_center'] * h)
                results['object_positions'].append((abs_x, abs_y))
                results['center_positions'].append((float(label['x_center']), float(label['y_center'])))

        # 통계 계산
        stats_results: Dict[str, Any] = {}

        # 클래스 분포
        if results['class_distribution']:
            class_counts = pd.Series(results['class_distribution']).value_counts().to_dict()
            # numpy int 스칼라 방지
            class_counts = {int(k): int(v) for k, v in class_counts.items()}
            stats_results['class_distribution'] = class_counts
        else:
            stats_results['class_distribution'] = {}

        # 수치형 통계
        for key in ['object_sizes', 'objects_per_image', 'aspect_ratios']:
            if results[key]:
                arr = np.asarray(results[key], dtype=np.float64)
                stats_results[key] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'median': float(np.median(arr))
                }

        # 중심 좌표 분포
        if results['center_positions']:
            center_x = np.asarray([p[0] for p in results['center_positions']], dtype=np.float64)
            center_y = np.asarray([p[1] for p in results['center_positions']], dtype=np.float64)

            stats_results['center_position_x'] = {
                'mean': float(np.mean(center_x)),
                'std': float(np.std(center_x)),
                'distribution': np.histogram(center_x, bins=10, range=(0, 1))[0].astype(int).tolist()
            }
            stats_results['center_position_y'] = {
                'mean': float(np.mean(center_y)),
                'std': float(np.std(center_y)),
                'distribution': np.histogram(center_y, bins=10, range=(0, 1))[0].astype(int).tolist()
            }

        self.analysis_results[dataset_name]['object_properties'] = stats_results
        return results

    # ----------------------------
    # 배경 복잡도 분석
    # ----------------------------
    def analyze_background_complexity(self, images: List[np.ndarray], dataset_name: str):
        results = {
            'edge_density': [],
            'texture_complexity': [],
            'color_diversity': [],
            'gradient_magnitude': []
        }

        print(f"Analyzing background complexity for {dataset_name}...")

        for i, image in enumerate(images):
            if i % 50 == 0:
                print(f"  Processing image {i+1}/{len(images)}")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 엣지 밀도
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.sum(edges > 0) / edges.size)
            results['edge_density'].append(edge_density)

            # 텍스처 복잡도 (LBP)
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            texture_complexity = int(len(np.unique(lbp)))
            results['texture_complexity'].append(texture_complexity)

            # 색상 다양성
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            unique_colors = int(len(np.unique(hsv.reshape(-1, hsv.shape[2]), axis=0)))
            color_diversity = float(unique_colors / (image.shape[0] * image.shape[1]))
            results['color_diversity'].append(color_diversity)

            # 그라디언트 크기
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2)))
            results['gradient_magnitude'].append(gradient_magnitude)

        # 통계
        stats_results: Dict[str, Any] = {}
        for key, values in results.items():
            if len(values) == 0:
                continue
            arr = np.asarray(values, dtype=np.float64)
            stats_results[key] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr))
            }

        self.analysis_results[dataset_name]['background_complexity'] = stats_results
        return results

    # ----------------------------
    # 통계적 비교
    # ----------------------------
    def statistical_comparison(self):
        """두 데이터셋 간 메트릭 평균 비교 (상대차 기반)"""
        print("Performing statistical comparison...")

        comparison_results: Dict[str, Dict[str, Any]] = {}

        d1 = self.analysis_results.get('dataset1', {})
        d2 = self.analysis_results.get('dataset2', {})

        categories = ['image_properties', 'object_properties', 'background_complexity']

        for category in categories:
            if category not in d1 or category not in d2:
                continue

            comparison_results[category] = {}

            for metric, m1 in d1[category].items():
                m2 = d2[category].get(metric)
                if not isinstance(m1, dict) or not isinstance(m2, dict):
                    continue
                if 'mean' not in m1 or 'mean' not in m2:
                    continue

                val1 = float(m1['mean'])
                val2 = float(m2['mean'])
                abs_diff = abs(val1 - val2)
                denom = max(abs(val1), abs(val2), 1e-8)
                rel_diff = float(abs_diff / denom * 100.0)

                # 20% 이상 차이면 유의미로 마킹
                sig = bool(rel_diff > 20.0)

                comparison_results[category][metric] = {
                    'dataset1_mean': val1,
                    'dataset2_mean': val2,
                    'absolute_difference': abs_diff,
                    'relative_difference_percent': rel_diff,
                    'significant_difference': sig
                }

        self.analysis_results['comparison'] = comparison_results
        return comparison_results

    # ----------------------------
    # 시각화
    # ----------------------------
    def create_visualizations(self, data1: Dict[str, Any], data2: Dict[str, Any]):
        print("Creating visualizations...")

        plt.style.use('seaborn-v0_8')
        fig_size = (15, 10)

        # 1) 이미지 속성
        fig, axes = plt.subplots(2, 3, figsize=fig_size)
        fig.suptitle('Image Properties Comparison', fontsize=16)

        metrics = ['brightness', 'contrast', 'saturation', 'resolution', 'aspect_ratio', 'blur_metric']
        for i, metric in enumerate(metrics):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            if metric in data1 and metric in data2 and len(data1[metric]) and len(data2[metric]):
                ax.hist(data1[metric], alpha=0.7, label='Dataset 1', bins=30)
                ax.hist(data2[metric], alpha=0.7, label='Dataset 2', bins=30)
                ax.set_title(metric.replace('_', ' ').title())
                ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'image_properties_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2) 객체 크기/개수
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        if 'object_sizes' in data1 and 'object_sizes' in data2 and len(data1['object_sizes']) and len(data2['object_sizes']):
            plt.hist(data1['object_sizes'], alpha=0.7, label='Dataset 1', bins=30)
            plt.hist(data2['object_sizes'], alpha=0.7, label='Dataset 2', bins=30)
            plt.xlabel('Object Size (pixels²)')
            plt.ylabel('Frequency')
            plt.title('Object Size Distribution')
            plt.legend()
            plt.yscale('log')

        plt.subplot(1, 2, 2)
        if 'objects_per_image' in data1 and 'objects_per_image' in data2 and len(data1['objects_per_image']) and len(data2['objects_per_image']):
            plt.hist(data1['objects_per_image'], alpha=0.7, label='Dataset 1', bins=20)
            plt.hist(data2['objects_per_image'], alpha=0.7, label='Dataset 2', bins=20)
            plt.xlabel('Objects per Image')
            plt.ylabel('Frequency')
            plt.title('Objects per Image Distribution')
            plt.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'object_analysis_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3) 배경 복잡도
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Background Complexity Comparison', fontsize=16)

        complexity_metrics = ['edge_density', 'texture_complexity', 'color_diversity', 'gradient_magnitude']
        for i, metric in enumerate(complexity_metrics):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            if metric in data1 and metric in data2 and len(data1[metric]) and len(data2[metric]):
                ax.hist(data1[metric], alpha=0.7, label='Dataset 1', bins=30)
                ax.hist(data2[metric], alpha=0.7, label='Dataset 2', bins=30)
                ax.set_title(metric.replace('_', ' ').title())
                ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'background_complexity_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4) 색조 분포
        if 'hue_distribution' in data1 and 'hue_distribution' in data2 and len(data1['hue_distribution']) and len(data2['hue_distribution']):
            plt.figure(figsize=(12, 6))
            hue_bins = np.arange(36) * 5  # 0-180도를 36개 빈
            mean_hue1 = np.mean(np.stack(data1['hue_distribution'], axis=0), axis=0)
            mean_hue2 = np.mean(np.stack(data2['hue_distribution'], axis=0), axis=0)

            plt.plot(hue_bins, mean_hue1, label='Dataset 1', linewidth=2)
            plt.plot(hue_bins, mean_hue2, label='Dataset 2', linewidth=2)
            plt.xlabel('Hue (degrees)')
            plt.ylabel('Average Frequency')
            plt.title('Hue Distribution Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'hue_distribution_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

    # ----------------------------
    # JSON 직렬화 변환기 + default
    # ----------------------------
    def convert_numpy_types(self, obj: Any):
        """numpy/pandas 스칼라와 배열, tuple/set까지 JSON 직렬화 가능한 형태로 변환"""
        # dict
        if isinstance(obj, dict):
            return {str(self.convert_numpy_types(k)): self.convert_numpy_types(v) for k, v in obj.items()}
        # list/tuple/set -> list
        if isinstance(obj, (list, tuple, set)):
            return [self.convert_numpy_types(x) for x in obj]
        # numpy 배열
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # numpy 스칼라들
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        # 기본 타입
        return obj

    @staticmethod
    def json_default(o: Any):
        """json.dump용 백업 변환기(이중 안전망)"""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (tuple, set)):
            return list(o)
        return str(o)

    # ----------------------------
    # 리포트 생성
    # ----------------------------
    def generate_report(self):
        print("Generating analysis report...")

        report: Dict[str, Any] = {
            'summary': {},
            'key_differences': [],
            'recommendations': []
        }

        comparison = self.analysis_results.get('comparison', {})
        significant_differences = []

        for category, metrics in comparison.items():
            for metric, data in metrics.items():
                if isinstance(data, dict) and bool(data.get('significant_difference', False)):
                    significant_differences.append({
                        'category': category,
                        'metric': metric,
                        'dataset1_value': float(data.get('dataset1_mean', np.nan)),
                        'dataset2_value': float(data.get('dataset2_mean', np.nan)),
                        'relative_difference': float(data.get('relative_difference_percent', np.nan))
                    })

        # 상대적 차이 내림차순
        significant_differences.sort(key=lambda x: x['relative_difference'], reverse=True)
        report['key_differences'] = significant_differences

        # 권장사항 (상위 5개 기준 룰 기반)
        recommendations: List[str] = []
        for diff in significant_differences[:5]:
            metric = diff['metric']
            category = diff['category']

            if category == 'image_properties':
                if metric == 'brightness':
                    recommendations.append("조명 차이가 큼: adaptive brightness/contrast, histogram equalization/CLAHE 적용 권장")
                elif metric == 'contrast':
                    recommendations.append("대비 차이가 큼: CLAHE 및 gamma/contrast augmentation 강화")
                elif metric == 'saturation':
                    recommendations.append("채도 차이가 큼: HSV/color jitter 파라미터 상향 및 컬러 정규화 검토")

            elif category == 'object_properties':
                if metric == 'object_sizes':
                    recommendations.append("객체 크기 분포 차이: multi-scale training, scale jitter 확대, imgsz↑ 고려")
                elif metric == 'objects_per_image':
                    recommendations.append("이미지당 객체 수 차이: mosaic/copy-paste augmentation 적용")

            elif category == 'background_complexity':
                if metric == 'edge_density':
                    recommendations.append("배경 복잡도 차이: background randomization/occlusion/shadow augmentation 검토")

        report['recommendations'] = recommendations

        avg_rel = float(np.mean([d['relative_difference'] for d in significant_differences])) if significant_differences else 0.0
        most_diff = significant_differences[0] if significant_differences else None

        report['summary'] = {
            'total_significant_differences': int(len(significant_differences)),
            'most_different_metric': most_diff,
            'average_relative_difference': float(avg_rel)
        }

        # JSON 저장 (변환 + default 백업)
        converted_report = self.convert_numpy_types(report)
        with open(self.output_dir / 'analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(converted_report, f, indent=2, ensure_ascii=False, default=self.json_default)

        # TXT 리포트
        with open(self.output_dir / 'analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write("Dataset Comparison Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"총 유의미한 차이점: {report['summary']['total_significant_differences']}개\n")
            f.write(f"평균 상대적 차이: {report['summary']['average_relative_difference']:.2f}%\n\n")

            f.write("주요 차이점 (상위 10개):\n")
            f.write("-" * 30 + "\n")
            for i, diff in enumerate(significant_differences[:10], 1):
                f.write(f"{i}. {diff['category']} - {diff['metric']}\n")
                f.write(f"   Dataset 1: {diff['dataset1_value']:.4f}\n")
                f.write(f"   Dataset 2: {diff['dataset2_value']:.4f}\n")
                f.write(f"   상대적 차이: {diff['relative_difference']:.2f}%\n\n")

            f.write("권장사항:\n")
            f.write("-" * 30 + "\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n\n")

        return report


def parse_args():
    p = argparse.ArgumentParser(description="Pepper(train) vs Pepper(test) EDA with safe JSON serialization")
    p.add_argument("--d1-img", default="/data/ioCrops/pepper/dataset/train_v1.1_fruit/images/train")
    p.add_argument("--d1-lbl", default="/data/ioCrops/pepper/dataset/train_v1.1_fruit/labels/train")
    p.add_argument("--d2-img", default="/home/cat123/yolov8-fruit_detection/yolo_dataset_new/pepper/images/test")
    p.add_argument("--d2-lbl", default="/home/cat123/yolov8-fruit_detection/yolo_dataset_new/pepper/labels/test")
    p.add_argument("--out",    default="/home/cat123/yolov8-fruit_detection/analysis_results")
    p.add_argument("--samples", type=int, default=200, help="각 데이터셋에서 로드할 샘플 수")
    p.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return p.parse_args()


def main():
    args = parse_args()

    analyzer = DatasetAnalyzer(args.out)

    # 랜덤 시드 고정
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Starting dataset analysis...")
    print(f"Results will be saved to: {analyzer.output_dir}")

    # 1) 데이터 로드
    print("\n1. Loading datasets...")
    images1, labels1, paths1 = analyzer.load_random_samples(args.d1_img, args.d1_lbl, args.samples)
    images2, labels2, paths2 = analyzer.load_random_samples(args.d2_img, args.d2_lbl, args.samples)

    if len(images1) == 0 or len(images2) == 0:
        print("Error: Failed to load images from one or both datasets")
        return

    # 2) 이미지 속성
    print("\n2. Analyzing image properties...")
    img_data1 = analyzer.analyze_image_properties(images1, 'dataset1')
    img_data2 = analyzer.analyze_image_properties(images2, 'dataset2')

    # 3) 객체 속성
    print("\n3. Analyzing object properties...")
    obj_data1 = analyzer.analyze_object_properties(labels1, images1, 'dataset1')
    obj_data2 = analyzer.analyze_object_properties(labels2, images2, 'dataset2')

    # 4) 배경 복잡도
    print("\n4. Analyzing background complexity...")
    bg_data1 = analyzer.analyze_background_complexity(images1, 'dataset1')
    bg_data2 = analyzer.analyze_background_complexity(images2, 'dataset2')

    # 5) 통계 비교
    print("\n5. Performing statistical comparison...")
    comparison = analyzer.statistical_comparison()

    # 6) 시각화
    print("\n6. Creating visualizations...")
    combined_data1 = {**img_data1, **obj_data1, **bg_data1}
    combined_data2 = {**img_data2, **obj_data2, **bg_data2}
    analyzer.create_visualizations(combined_data1, combined_data2)

    # 7) 보고서
    print("\n7. Generating report...")
    report = analyzer.generate_report()

    # 8) 전체 결과 JSON 저장 (안전 변환 + default)
    converted_results = analyzer.convert_numpy_types(analyzer.analysis_results)
    with open(analyzer.output_dir / 'complete_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(converted_results, f, indent=2, ensure_ascii=False, default=analyzer.json_default)

    print(f"\nAnalysis completed! Results saved to: {analyzer.output_dir}")
    print(f"Found {report['summary']['total_significant_differences']} significant differences between datasets")

    if report['summary']['most_different_metric']:
        most_diff = report['summary']['most_different_metric']
        print(f"Most significant difference: {most_diff['metric']} ({most_diff['relative_difference']:.2f}% difference)")


if __name__ == "__main__":
    main()
