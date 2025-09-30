import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
import json
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import warnings
warnings.filterwarnings('ignore')

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ComprehensiveEDA:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 저장용
        self.datasets = {}
        self.analysis_results = {}
        
        # 클래스 매핑
        self.class_names = {0: 'ripened', 1: 'ripening', 2: 'unripened'}
        
    def load_dataset(self, name, image_dir, label_dir, n_samples=None):
        """데이터셋 로드"""
        print(f"Loading {name} dataset...")
        
        image_files = list(Path(image_dir).glob('*.png'))
        
        if n_samples and len(image_files) > n_samples:
            image_files = random.sample(image_files, n_samples)
            
        dataset = {
            'images': [],
            'labels': [],
            'image_paths': [],
            'image_features': [],
            'object_features': []
        }
        
        for img_path in image_files:
            # 이미지 로드
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            # 라벨 로드
            label_path = Path(label_dir) / (img_path.stem + '.txt')
            objects = []
            
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, x_center, y_center, width, height = map(float, parts)
                            objects.append({
                                'class_id': int(class_id),
                                'x_center': x_center,
                                'y_center': y_center,
                                'width': width,
                                'height': height
                            })
            
            # 이미지 특징 추출
            img_features = self.extract_image_features(image)
            obj_features = self.extract_object_features(objects, image.shape[:2])
            
            dataset['images'].append(image)
            dataset['labels'].append(objects)
            dataset['image_paths'].append(str(img_path))
            dataset['image_features'].append(img_features)
            dataset['object_features'].append(obj_features)
        
        self.datasets[name] = dataset
        print(f"Loaded {len(dataset['images'])} images for {name}")
        return dataset
    
    def extract_image_features(self, image):
        """이미지에서 특징 추출"""
        h, w = image.shape[:2]
        
        # 색상 공간 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        features = {
            # 기본 속성
            'width': w,
            'height': h,
            'aspect_ratio': w / h,
            'area': w * h,
            
            # 색상 특징
            'mean_brightness': np.mean(hsv[:,:,2]),
            'std_brightness': np.std(hsv[:,:,2]),
            'mean_saturation': np.mean(hsv[:,:,1]),
            'std_saturation': np.std(hsv[:,:,1]),
            'mean_hue': np.mean(hsv[:,:,0]),
            'std_hue': np.std(hsv[:,:,0]),
            
            # LAB 색공간
            'mean_l': np.mean(lab[:,:,0]),
            'mean_a': np.mean(lab[:,:,1]),
            'mean_b': np.mean(lab[:,:,2]),
            
            # 텍스처 특징
            'contrast': cv2.Laplacian(gray, cv2.CV_64F).var(),
            'edge_density': np.sum(cv2.Canny(gray, 50, 150) > 0) / (w * h),
            
            # 통계적 특징
            'entropy': self.calculate_entropy(gray),
            'std_intensity': np.std(gray),
            'skewness': stats.skew(gray.flatten()),
            'kurtosis': stats.kurtosis(gray.flatten()),
        }
        
        # 색상 히스토그램
        hist_b = cv2.calcHist([image], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [32], [0, 256])
        
        features.update({
            f'hist_b_{i}': hist_b[i][0] for i in range(32)
        })
        features.update({
            f'hist_g_{i}': hist_g[i][0] for i in range(32)
        })
        features.update({
            f'hist_r_{i}': hist_r[i][0] for i in range(32)
        })
        
        return features
    
    def extract_object_features(self, objects, image_shape):
        """객체에서 특징 추출"""
        h, w = image_shape
        
        features = {
            'num_objects': len(objects),
            'object_density': len(objects) / (w * h) * 1000000,  # per million pixels
        }
        
        if objects:
            # 객체 크기 분석
            areas = [obj['width'] * obj['height'] * w * h for obj in objects]
            features.update({
                'mean_object_area': np.mean(areas),
                'std_object_area': np.std(areas),
                'min_object_area': np.min(areas),
                'max_object_area': np.max(areas),
                'total_object_area': np.sum(areas),
                'object_coverage': np.sum(areas) / (w * h),
            })
            
            # 위치 분석
            x_centers = [obj['x_center'] for obj in objects]
            y_centers = [obj['y_center'] for obj in objects]
            
            features.update({
                'mean_x_position': np.mean(x_centers),
                'std_x_position': np.std(x_centers),
                'mean_y_position': np.mean(y_centers),
                'std_y_position': np.std(y_centers),
            })
            
            # 클래스 분포
            class_counts = pd.Series([obj['class_id'] for obj in objects]).value_counts()
            for class_id in [0, 1, 2]:  # ripened, ripening, unripened
                features[f'count_class_{class_id}'] = class_counts.get(class_id, 0)
                features[f'ratio_class_{class_id}'] = class_counts.get(class_id, 0) / len(objects)
        else:
            # 객체가 없는 경우 기본값
            zero_features = [
                'mean_object_area', 'std_object_area', 'min_object_area', 'max_object_area',
                'total_object_area', 'object_coverage', 'mean_x_position', 'std_x_position',
                'mean_y_position', 'std_y_position'
            ]
            for feat in zero_features:
                features[feat] = 0
                
            for class_id in [0, 1, 2]:
                features[f'count_class_{class_id}'] = 0
                features[f'ratio_class_{class_id}'] = 0
        
        return features
    
    def calculate_entropy(self, image):
        """이미지 엔트로피 계산"""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist = hist[hist > 0]
        prob = hist / np.sum(hist)
        return -np.sum(prob * np.log2(prob))
    
    def create_dataframes(self):
        """분석을 위한 DataFrame 생성"""
        self.dataframes = {}
        
        for dataset_name, dataset in self.datasets.items():
            # 이미지 특징 DataFrame
            img_df = pd.DataFrame(dataset['image_features'])
            img_df['dataset'] = dataset_name
            img_df['image_idx'] = range(len(img_df))
            
            # 객체 특징 DataFrame
            obj_df = pd.DataFrame(dataset['object_features'])
            obj_df['dataset'] = dataset_name
            obj_df['image_idx'] = range(len(obj_df))
            
            # 개별 객체 DataFrame
            individual_objects = []
            for img_idx, objects in enumerate(dataset['labels']):
                for obj in objects:
                    obj_data = obj.copy()
                    obj_data['dataset'] = dataset_name
                    obj_data['image_idx'] = img_idx
                    obj_data['class_name'] = self.class_names[obj['class_id']]
                    individual_objects.append(obj_data)
            
            individual_df = pd.DataFrame(individual_objects) if individual_objects else pd.DataFrame()
            
            self.dataframes[dataset_name] = {
                'image_features': img_df,
                'object_features': obj_df,
                'individual_objects': individual_df
            }
    
    def statistical_analysis(self):
        """통계적 분석 수행"""
        print("Performing statistical analysis...")
        
        if len(self.datasets) < 2:
            print("Need at least 2 datasets for comparison")
            return
        
        dataset_names = list(self.datasets.keys())
        df1_img = self.dataframes[dataset_names[0]]['image_features']
        df2_img = self.dataframes[dataset_names[1]]['image_features']
        
        # 수치형 컬럼만 선택
        numeric_cols = df1_img.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['dataset', 'image_idx']]
        
        comparison_results = {}
        
        for col in numeric_cols:
            if col in df1_img.columns and col in df2_img.columns:
                val1 = df1_img[col].dropna()
                val2 = df2_img[col].dropna()
                
                if len(val1) > 0 and len(val2) > 0:
                    # Kolmogorov-Smirnov 테스트
                    ks_stat, ks_p = ks_2samp(val1, val2)
                    
                    # t-test
                    t_stat, t_p = stats.ttest_ind(val1, val2)
                    
                    # 효과 크기 (Cohen's d)
                    pooled_std = np.sqrt(((len(val1)-1)*val1.var() + (len(val2)-1)*val2.var()) / (len(val1)+len(val2)-2))
                    cohens_d = (val1.mean() - val2.mean()) / pooled_std if pooled_std > 0 else 0
                    
                    comparison_results[col] = {
                        'dataset1_mean': val1.mean(),
                        'dataset1_std': val1.std(),
                        'dataset2_mean': val2.mean(),
                        'dataset2_std': val2.std(),
                        'ks_statistic': ks_stat,
                        'ks_p_value': ks_p,
                        't_statistic': t_stat,
                        't_p_value': t_p,
                        'cohens_d': cohens_d,
                        'effect_size': self.interpret_cohens_d(cohens_d)
                    }
        
        self.analysis_results['statistical_comparison'] = comparison_results
        return comparison_results
    
    def interpret_cohens_d(self, d):
        """Cohen's d 해석"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def correlation_analysis(self):
        """상관관계 분석"""
        print("Performing correlation analysis...")
        
        correlations = {}
        
        for dataset_name, dfs in self.dataframes.items():
            img_df = dfs['image_features']
            numeric_cols = img_df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['dataset', 'image_idx']]
            
            if len(numeric_cols) > 1:
                corr_matrix = img_df[numeric_cols].corr()
                correlations[dataset_name] = corr_matrix
        
        self.analysis_results['correlations'] = correlations
        return correlations
    
    def outlier_detection(self):
        """이상치 탐지"""
        print("Detecting outliers...")
        
        outliers = {}
        
        for dataset_name, dfs in self.dataframes.items():
            img_df = dfs['image_features']
            numeric_cols = img_df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['dataset', 'image_idx']]
            
            dataset_outliers = {}
            
            for col in numeric_cols:
                Q1 = img_df[col].quantile(0.25)
                Q3 = img_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_indices = img_df[(img_df[col] < lower_bound) | (img_df[col] > upper_bound)].index
                
                dataset_outliers[col] = {
                    'count': len(outlier_indices),
                    'percentage': len(outlier_indices) / len(img_df) * 100,
                    'indices': outlier_indices.tolist()
                }
            
            outliers[dataset_name] = dataset_outliers
        
        self.analysis_results['outliers'] = outliers
        return outliers
    
    def class_distribution_analysis(self):
        """클래스 분포 분석"""
        print("Analyzing class distributions...")
        
        class_analysis = {}
        
        for dataset_name, dfs in self.dataframes.items():
            if not dfs['individual_objects'].empty:
                class_counts = dfs['individual_objects']['class_id'].value_counts().sort_index()
                total_objects = len(dfs['individual_objects'])
                
                class_analysis[dataset_name] = {
                    'class_counts': class_counts.to_dict(),
                    'class_percentages': (class_counts / total_objects * 100).to_dict(),
                    'total_objects': total_objects,
                    'images_with_objects': dfs['object_features']['num_objects'].gt(0).sum(),
                    'average_objects_per_image': dfs['object_features']['num_objects'].mean()
                }
        
        self.analysis_results['class_distribution'] = class_analysis
        return class_analysis
    
    def dimensionality_reduction(self):
        """차원 축소 및 클러스터링"""
        print("Performing dimensionality reduction...")
        
        # 모든 데이터셋 결합
        all_img_features = []
        dataset_labels = []
        
        for dataset_name, dfs in self.dataframes.items():
            img_df = dfs['image_features']
            numeric_cols = img_df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['dataset', 'image_idx']]
            
            features = img_df[numeric_cols].fillna(0)
            all_img_features.append(features)
            dataset_labels.extend([dataset_name] * len(features))
        
        if len(all_img_features) > 1:
            combined_features = pd.concat(all_img_features, ignore_index=True)
            
            # 표준화
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(combined_features)
            
            # PCA
            pca = PCA(n_components=min(10, scaled_features.shape[1]))
            pca_result = pca.fit_transform(scaled_features)
            
            # K-means 클러스터링
            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            self.analysis_results['dimensionality_reduction'] = {
                'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
                'pca_components': pca_result,
                'cluster_labels': cluster_labels,
                'dataset_labels': dataset_labels
            }
    
    def create_comprehensive_visualizations(self):
        """종합적인 시각화 생성"""
        print("Creating comprehensive visualizations...")
        
        # 1. 데이터셋 비교 대시보드
        self.create_comparison_dashboard()
        
        # 2. 클래스 분포 시각화
        self.create_class_distribution_plots()
        
        # 3. 상관관계 히트맵
        self.create_correlation_heatmaps()
        
        # 4. 분포 비교 플롯
        self.create_distribution_comparison()
        
        # 5. 이상치 시각화
        self.create_outlier_plots()
        
        # 6. PCA 시각화
        self.create_pca_plots()
    
    def create_comparison_dashboard(self):
        """비교 대시보드 생성"""
        if len(self.datasets) < 2:
            return
            
        dataset_names = list(self.datasets.keys())
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Brightness', 'Contrast', 'Saturation', 
                          'Object Density', 'Object Coverage', 'Image Size'],
            specs=[[{"secondary_y": False}]*3]*2
        )
        
        metrics = ['mean_brightness', 'contrast', 'mean_saturation', 
                  'object_density', 'object_coverage', 'area']
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, metric in enumerate(metrics):
            row = i // 3 + 1
            col = i % 3 + 1
            
            for j, dataset_name in enumerate(dataset_names[:4]):
                df_img = self.dataframes[dataset_name]['image_features']
                df_obj = self.dataframes[dataset_name]['object_features']
                
                if metric in df_img.columns:
                    data = df_img[metric]
                elif metric in df_obj.columns:
                    data = df_obj[metric]
                else:
                    continue
                
                fig.add_trace(
                    go.Histogram(x=data, name=f'{dataset_name}_{metric}', 
                               opacity=0.7, marker_color=colors[j]),
                    row=row, col=col
                )
        
        fig.update_layout(height=800, title_text="Dataset Comparison Dashboard")
        fig.write_html(self.output_dir / 'comparison_dashboard.html')
    
    def create_class_distribution_plots(self):
        """클래스 분포 시각화"""
        if 'class_distribution' not in self.analysis_results:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Class Distribution Analysis', fontsize=16)
        
        dataset_names = list(self.analysis_results['class_distribution'].keys())
        
        # 클래스 개수 비교
        ax1 = axes[0, 0]
        class_data = []
        for dataset_name in dataset_names:
            for class_id, count in self.analysis_results['class_distribution'][dataset_name]['class_counts'].items():
                class_data.append({
                    'Dataset': dataset_name,
                    'Class': self.class_names[class_id],
                    'Count': count
                })
        
        if class_data:
            class_df = pd.DataFrame(class_data)
            sns.barplot(data=class_df, x='Class', y='Count', hue='Dataset', ax=ax1)
            ax1.set_title('Object Count by Class')
        
        # 클래스 비율 비교
        ax2 = axes[0, 1]
        if class_data:
            percentage_data = []
            for dataset_name in dataset_names:
                for class_id, percentage in self.analysis_results['class_distribution'][dataset_name]['class_percentages'].items():
                    percentage_data.append({
                        'Dataset': dataset_name,
                        'Class': self.class_names[class_id],
                        'Percentage': percentage
                    })
            
            if percentage_data:
                perc_df = pd.DataFrame(percentage_data)
                sns.barplot(data=perc_df, x='Class', y='Percentage', hue='Dataset', ax=ax2)
                ax2.set_title('Class Distribution (%)')
        
        # 이미지당 객체 수 분포
        ax3 = axes[1, 0]
        for dataset_name in dataset_names:
            df_obj = self.dataframes[dataset_name]['object_features']
            ax3.hist(df_obj['num_objects'], alpha=0.7, label=dataset_name, bins=20)
        ax3.set_xlabel('Objects per Image')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Objects per Image Distribution')
        ax3.legend()
        
        # 객체 크기 분포
        ax4 = axes[1, 1]
        for dataset_name in dataset_names:
            df_obj = self.dataframes[dataset_name]['individual_objects']
            if not df_obj.empty:
                areas = df_obj['width'] * df_obj['height']
                ax4.hist(areas, alpha=0.7, label=dataset_name, bins=30)
        ax4.set_xlabel('Object Size (normalized)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Object Size Distribution')
        ax4.legend()
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_correlation_heatmaps(self):
        """상관관계 히트맵 생성"""
        if 'correlations' not in self.analysis_results:
            return
            
        n_datasets = len(self.analysis_results['correlations'])
        fig, axes = plt.subplots(1, n_datasets, figsize=(8*n_datasets, 6))
        
        if n_datasets == 1:
            axes = [axes]
        
        for i, (dataset_name, corr_matrix) in enumerate(self.analysis_results['correlations'].items()):
            # 주요 특징만 선택 (너무 많으면 읽기 어려움)
            important_features = ['mean_brightness', 'contrast', 'mean_saturation', 'object_density', 
                                'object_coverage', 'num_objects', 'aspect_ratio', 'edge_density']
            
            available_features = [f for f in important_features if f in corr_matrix.columns]
            
            if available_features:
                subset_corr = corr_matrix.loc[available_features, available_features]
                
                sns.heatmap(subset_corr, annot=True, cmap='coolwarm', center=0,
                           fmt='.2f', ax=axes[i], cbar_kws={'shrink': 0.8})
                axes[i].set_title(f'{dataset_name} Feature Correlations')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_distribution_comparison(self):
        """분포 비교 플롯"""
        if len(self.datasets) < 2:
            return
            
        key_features = ['mean_brightness', 'contrast', 'mean_saturation', 'object_density', 
                       'num_objects', 'object_coverage']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Distribution Comparison', fontsize=16)
        
        dataset_names = list(self.datasets.keys())
        
        for i, feature in enumerate(key_features):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            for dataset_name in dataset_names:
                # 이미지 특징에서 찾기
                img_df = self.dataframes[dataset_name]['image_features']
                obj_df = self.dataframes[dataset_name]['object_features']
                
                if feature in img_df.columns:
                    data = img_df[feature].dropna()
                elif feature in obj_df.columns:
                    data = obj_df[feature].dropna()
                else:
                    continue
                
                if len(data) > 0:
                    sns.histplot(data, alpha=0.7, label=dataset_name, ax=ax, kde=True)
            
            ax.set_title(f'{feature.replace("_", " ").title()}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_outlier_plots(self):
        """이상치 시각화"""
        if 'outliers' not in self.analysis_results:
            return
            
        # 이상치 비율 비교
        outlier_summary = []
        for dataset_name, outliers in self.analysis_results['outliers'].items():
            for feature, data in outliers.items():
                outlier_summary.append({
                    'Dataset': dataset_name,
                    'Feature': feature,
                    'Outlier_Percentage': data['percentage']
                })
        
        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary)
            
            plt.figure(figsize=(15, 8))
            
            # 상위 20개 특징만 표시
            top_features = outlier_df.groupby('Feature')['Outlier_Percentage'].mean().nlargest(20).index
            outlier_subset = outlier_df[outlier_df['Feature'].isin(top_features)]
            
            sns.barplot(data=outlier_subset, x='Feature', y='Outlier_Percentage', hue='Dataset')
            plt.xticks(rotation=45, ha='right')
            plt.title('Outlier Percentage by Feature')
            plt.ylabel('Outlier Percentage (%)')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'outlier_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_pca_plots(self):
        """PCA 시각화"""
        if 'dimensionality_reduction' not in self.analysis_results:
            return
            
        pca_data = self.analysis_results['dimensionality_reduction']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA 설명된 분산
        axes[0].bar(range(1, len(pca_data['pca_explained_variance'])+1), 
                   pca_data['pca_explained_variance'])
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('PCA Explained Variance')
        
        # PCA 스캐터 플롯
        pca_components = np.array(pca_data['pca_components'])
        dataset_labels = pca_data['dataset_labels']
        
        unique_datasets = list(set(dataset_labels))
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, dataset in enumerate(unique_datasets):
            mask = [label == dataset for label in dataset_labels]
            axes[1].scatter(pca_components[mask, 0], pca_components[mask, 1], 
                          c=colors[i % len(colors)], label=dataset, alpha=0.7)
        
        axes[1].set_xlabel('First Principal Component')
        axes[1].set_ylabel('Second Principal Component')
        axes[1].set_title('PCA Visualization')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self):
        """종합 보고서 생성"""
        print("Generating comprehensive EDA report...")
        
        report = {
            'dataset_summary': {},
            'statistical_findings': {},
            'key_insights': [],
            'recommendations': []
        }
        
        # 데이터셋 요약
        for dataset_name, dataset in self.datasets.items():
            report['dataset_summary'][dataset_name] = {
                'total_images': len(dataset['images']),
                'total_objects': sum(len(labels) for labels in dataset['labels']),
                'average_objects_per_image': np.mean([len(labels) for labels in dataset['labels']]),
                'class_distribution': self.analysis_results.get('class_distribution', {}).get(dataset_name, {})
            }
        
        # 통계적 발견사항
        if 'statistical_comparison' in self.analysis_results:
            significant_differences = []
            for feature, stats in self.analysis_results['statistical_comparison'].items():
                if stats['ks_p_value'] < 0.05 and stats['effect_size'] in ['medium', 'large']:
                    significant_differences.append({
                        'feature': feature,
                        'effect_size': stats['effect_size'],
                        'cohens_d': stats['cohens_d'],
                        'p_value': stats['ks_p_value']
                    })
            
            report['statistical_findings'] = {
                'significant_differences': significant_differences,
                'total_features_compared': len(self.analysis_results['statistical_comparison'])
            }
        
        # 주요 인사이트 생성
        insights = []
        
        # 클래스 불균형 체크
        if 'class_distribution' in self.analysis_results:
            for dataset_name, class_data in self.analysis_results['class_distribution'].items():
                percentages = list(class_data['class_percentages'].values())
                if percentages and (max(percentages) > 70 or min(percentages) < 10):
                    insights.append(f"{dataset_name}에서 심각한 클래스 불균형 발견")
        
        # 이상치 문제
        if 'outliers' in self.analysis_results:
            for dataset_name, outliers in self.analysis_results['outliers'].items():
                high_outlier_features = [f for f, data in outliers.items() if data['percentage'] > 10]
                if high_outlier_features:
                    insights.append(f"{dataset_name}에서 {len(high_outlier_features)}개 특징에 높은 이상치 비율")
        
        report['key_insights'] = insights
        
        # 권장사항 생성
        recommendations = [
            "클래스 불균형 해결을 위한 weighted sampling 또는 SMOTE 적용",
            "이상치가 많은 특징에 대한 robust scaling 적용",
            "상관관계가 높은 특징들에 대한 차원 축소 고려",
            "도메인 간 분포 차이가 큰 특징에 대한 정규화 전략 수립"
        ]
        
        if 'statistical_comparison' in self.analysis_results:
            large_effect_features = [
                stats['feature'] for stats in report['statistical_findings']['significant_differences']
                if stats['effect_size'] == 'large'
            ]
            if large_effect_features:
                recommendations.append(f"큰 효과 크기를 보인 특징들({', '.join(large_effect_features[:3])})에 대한 도메인 적응 필요")
        
        report['recommendations'] = recommendations
        
        # JSON 저장
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        converted_report = convert_types(report)
        converted_results = convert_types(self.analysis_results)
        
        with open(self.output_dir / 'comprehensive_eda_report.json', 'w', encoding='utf-8') as f:
            json.dump(converted_report, f, indent=2, ensure_ascii=False)
        
        with open(self.output_dir / 'full_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)
        
        # 텍스트 보고서
        with open(self.output_dir / 'eda_summary_report.txt', 'w', encoding='utf-8') as f:
            f.write("Comprehensive EDA Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Dataset Summary:\n")
            for dataset_name, summary in report['dataset_summary'].items():
                f.write(f"\n{dataset_name}:\n")
                f.write(f"  - Images: {summary['total_images']}\n")
                f.write(f"  - Objects: {summary['total_objects']}\n")
                f.write(f"  - Avg objects/image: {summary['average_objects_per_image']:.2f}\n")
            
            f.write(f"\nStatistical Findings:\n")
            if report['statistical_findings']:
                f.write(f"  - Features compared: {report['statistical_findings']['total_features_compared']}\n")
                f.write(f"  - Significant differences: {len(report['statistical_findings']['significant_differences'])}\n")
            
            f.write(f"\nKey Insights:\n")
            for insight in report['key_insights']:
                f.write(f"  - {insight}\n")
            
            f.write(f"\nRecommendations:\n")
            for rec in report['recommendations']:
                f.write(f"  - {rec}\n")
        
        return report

def main():
    # 출력 디렉토리
    output_dir = "/home/cat123/yolov8-fruit_detection/analysis_results"
    
    # EDA 분석기 초기화
    eda = ComprehensiveEDA(output_dir)
    
    # 랜덤 시드 설정
    random.seed(42)
    np.random.seed(42)
    
    print("Starting Comprehensive EDA...")
    
    # 데이터셋 로드
    datasets_to_load = [
        ("dataset1_pepper", "/data/ioCrops/pepper/dataset/train_v1.1_fruit/images/train", 
         "/data/ioCrops/pepper/dataset/train_v1.1_fruit/labels/train", 200),
        ("dataset2_pepper", "/home/cat123/yolov8-fruit_detection/yolo_dataset_new/pepper/images/test",
         "/home/cat123/yolov8-fruit_detection/yolo_dataset_new/pepper/labels/test", 200),
    ]
    
    # 추가 데이터셋들 (필요시 주석 해제)
    # ("dataset1_tomato", "/data/ioCrops/tomato/dataset/train_v1.3_fruit/images/train",
    #  "/data/ioCrops/tomato/dataset/train_v1.3_fruit/labels/train", 200),
    # ("dataset1_berry", "/data/ioCrops/berry/dataset/train_v1.0_fruit/images/train",
    #  "/data/ioCrops/berry/dataset/train_v1.0_fruit/labels/train", 200),
    
    for name, img_dir, label_dir, n_samples in datasets_to_load:
        eda.load_dataset(name, img_dir, label_dir, n_samples)
    
    if not eda.datasets:
        print("No datasets loaded successfully!")
        return
    
    # DataFrame 생성
    eda.create_dataframes()
    
    # 분석 수행
    eda.statistical_analysis()
    eda.correlation_analysis()
    eda.outlier_detection()
    eda.class_distribution_analysis()
    eda.dimensionality_reduction()
    
    # 시각화 생성
    eda.create_comprehensive_visualizations()
    
    # 종합 보고서 생성
    report = eda.generate_comprehensive_report()
    
    print(f"\nComprehensive EDA completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Key insights found: {len(report['key_insights'])}")

if __name__ == "__main__":
    main()