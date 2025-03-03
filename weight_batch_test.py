import torch
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
from collections import defaultdict
import cv2
from typing import List, Tuple, Dict
import queue
from yolov7.api_detect import API_Detect
from api_utils.utils_file import parse_cfg
import random


class BatchWeightModel:
    def __init__(self, batch_size=4):
        # 初始化配置
        self.cfg = parse_cfg('cfg.ini')
        self.batch_size = batch_size
        self.device = '0'

        # 初始化模型
        self.DZ_model = API_Detect(device=self.device, weights=self.cfg['DZ_model_detc'])

        # 预热模型
        self.warmup()

        self.rf_model = None
        self.gpu_lock = threading.Lock()
        self.timing_stats = defaultdict(list)

        # 批处理队列
        self.batch_queue = queue.Queue()
        self.result_dict = {}

        # 添加响应时间统计
        self.response_times = []

    def warmup(self):
        """模型预热"""
        print("Warming up model...")
        test_image = 'demo_imgs/T_K88061537_20221210135901.jpg'
        for _ in range(3):
            self.DZ_model.inference(test_image)

    def batch_detect_bbox(self, image_batch: List[str]) -> Tuple[List, float]:
        """批量检测边界框"""
        t1 = time.time()
        batch_results = []

        with self.gpu_lock:
            for img_path in image_batch:
                _, _, bbox_list = self.DZ_model.inference(img_path)
                batch_results.append(bbox_list)

        inference_time = time.time() - t1
        return batch_results, inference_time

    def bbox_to_features(self, bbox_list):
        """将边界框转换为特征向量 - 使用numpy优化"""
        t1 = time.time()
        if not bbox_list:
            return np.array([]), 0

        bbox_array = np.array(bbox_list)
        widths = bbox_array[:, 2] - bbox_array[:, 0]
        heights = bbox_array[:, 3] - bbox_array[:, 1]
        areas = widths * heights
        features = np.column_stack((widths, heights, areas))

        feature_time = time.time() - t1
        return features, feature_time

    def train(self, image_paths):
        """训练模型"""
        all_features = []
        all_weights = []

        for i in range(0, len(image_paths), self.batch_size):
            batch = image_paths[i:i + self.batch_size]
            bbox_lists, _ = self.batch_detect_bbox(batch)

            for bbox_list in bbox_lists:
                features, _ = self.bbox_to_features(bbox_list)
                if features.shape[0] > 0:
                    weights = [round(random.uniform(1, 3), 2) for _ in range(len(features))]
                    all_features.extend(features)
                    all_weights.extend(weights)

        X = np.array(all_features)
        y = np.array(all_weights)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.rf_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        self.rf_model.fit(X_train, y_train)
        return self.rf_model

    def process_single_request(self, image_path: str) -> Tuple[List[float], Dict]:
        """处理单个请求并记录响应时间"""
        start_time = time.time()

        # 执行检测
        bbox_lists, t1 = self.batch_detect_bbox([image_path])

        # 特征提取和预测
        features, t2 = self.bbox_to_features(bbox_lists[0])
        predictions = self.rf_model.predict(features) if features.shape[0] > 0 else []

        end_time = time.time()
        response_time = end_time - start_time

        # 记录响应时间
        self.response_times.append(response_time)

        timing = {
            't1': t1,
            't2': t2,
            'total': response_time
        }

        return predictions, timing


def calculate_concurrent_metrics(response_times: List[float]) -> Dict:
    """计算并发性能指标"""
    metrics = {
        'avg_response_time': np.mean(response_times),
        'max_response_time': np.max(response_times),
        'min_response_time': np.min(response_times),
        'p95_response_time': np.percentile(response_times, 95),
        'total_requests': len(response_times)
    }
    return metrics


def run_concurrent_test(model: BatchWeightModel, test_images: List[str], num_concurrent: int = 100) -> Dict:
    """运行并发测试"""
    # 确保有足够的测试图像
    if len(test_images) < num_concurrent:
        test_images = test_images * (num_concurrent // len(test_images) + 1)
    test_images = test_images[:num_concurrent]

    # 重置响应时间统计
    model.response_times = []

    # 创建线程池
    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        # 提交所有请求
        future_to_image = {
            executor.submit(model.process_single_request, image): image
            for image in test_images
        }

        # 等待所有请求完成
        for future in as_completed(future_to_image):
            try:
                predictions, timing = future.result()
            except Exception as e:
                print(f"处理请求时发生错误: {str(e)}")

    # 计算性能指标
    metrics = calculate_concurrent_metrics(model.response_times)
    return metrics


def main():
    # 初始化批处理模型
    batch_size = 4
    model = BatchWeightModel(batch_size=batch_size)

    # 训练模型
    train_images = ['demo_imgs/T_K88061537_20221210135901.jpg']
    model.train(train_images)

    # 准备测试数据
    test_images = [
        'demo_imgs/3_CH251930808_20221021090003.jpg',
        'demo_imgs/3_CH251930825_20221102090002.jpg',
        'demo_imgs/3_CH251930825_20221107170001.jpg',
        'demo_imgs/T_K88061537_20221210135901.jpg'
    ]

    # 运行100并发测试
    print("开始运行100并发测试...")
    metrics = run_concurrent_test(model, test_images, num_concurrent=100)

    # 打印性能指标
    print("\n性能测试结果:")
    print(f"100并发平均响应时间: {metrics['avg_response_time']:.3f} 秒")
    print(f"最大响应时间: {metrics['max_response_time']:.3f} 秒")
    print(f"最小响应时间: {metrics['min_response_time']:.3f} 秒")
    print(f"95百分位响应时间: {metrics['p95_response_time']:.3f} 秒")
    print(f"总请求数: {metrics['total_requests']}")


if __name__ == '__main__':
    main()