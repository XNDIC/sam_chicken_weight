import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import json
import math
from tqdm import tqdm
import sys

# 添加所需路径
current_dir = os.path.dirname(os.path.abspath(__file__))
yolov7_path = os.path.join(current_dir, 'yolov7')
sys.path.append(current_dir)
sys.path.append(yolov7_path)
sys.path.append(os.path.join(current_dir, 'efficientvit_master/efficientvit'))
sys.path.append(os.path.join(current_dir, 'efficientvit_master/segment_anything'))

'''

'''
class ChickenCounter:
    def __init__(self, model_cfg_path: str, device: str = '0'):
        """
        初始化鸡只计数器
        参数:
            model_cfg_path: 配置文件路径
            device: GPU设备号
        """
        try:
            # 确保models.yolo可以被正确导入
            import sys
            if yolov7_path not in sys.path:
                sys.path.append(yolov7_path)

            from yolov7.api_detect import API_Detect
            from api_utils.utils_file import parse_cfg
            from efficientvit_master.efficientvit.sam_model_zoo import create_sam_model
            from efficientvit_master.efficientvit.models.efficientvit.sam import EfficientViTSamPredictor

            # 初始化配置
            print("正在加载配置文件...")
            self.cfg = parse_cfg(model_cfg_path)

            # 初始化YOLOv7模型
            print("正在加载YOLOv7模型...")
            self.model = API_Detect(device=device, weights=self.cfg['DZ_model_detc'])

            # 初始化SAM模型
            print("正在加载SAM模型...")
            self.sam_model = create_sam_model(
                name="xl1",
                weight_url="efficientvit_master/assets/checkpoints/sam/xl1.pt"
            )
            self.sam_model = self.sam_model.cuda().eval()
            self.predictor = EfficientViTSamPredictor(self.sam_model)
            print("模型加载完成！")

        except ImportError as e:
            print(f"导入错误: {e}")
            print("Python路径:")
            for p in sys.path:
                print(f"  - {p}")
            raise
        except Exception as e:
            print(f"初始化错误: {e}")
            raise

    def bbox2sam_points(self, bbox_list: List, img_shape: Tuple) -> Tuple:
        """转换边界框为SAM点"""
        points, labels = [], []
        H, W = img_shape[:2]
        cx, cy = W // 2, H // 2
        max_distance = float('inf')
        best_c = None

        for bbox in bbox_list:
            x1, y1, x2, y2 = bbox
            x_c, y_c = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # 确保在圆形范围内
            distance = math.sqrt((x_c - cx) ** 2 + (y_c - cy) ** 2)
            if distance > H / 2:
                continue

            if distance < max_distance:
                max_distance = distance
                points = [(x_c, y_c), (x1, x2), (y1, y2)]
                labels = [1, 0, 0]
                best_c = [x_c, y_c]

        return points, labels, best_c

    def process_single_image(self, img_path: str) -> Dict:
        """
        处理单张图片，统计检测到的鸡只数量
        返回:
            {
                "success": bool,
                "detection_count": int,  # 检测到的鸡只数量
                "image_path": str,
                "error": str or None
            }
        """
        try:
            print(f"正在处理图片: {img_path}")

            # YOLO模型推理
            _, number, bbox_list = self.model.inference(img_path)
            detected_count = len(bbox_list)  # 检测到的鸡只数量

            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                return {
                    "success": False,
                    "detection_count": 0,
                    "image_path": img_path,
                    "error": "无法读取图片"
                }

            # 获取SAM点
            points, labels, bbox_c = self.bbox2sam_points(bbox_list, img.shape)

            if not points:
                return {
                    "success": False,
                    "detection_count": detected_count,
                    "image_path": img_path,
                    "error": "SAM处理失败"
                }

            # 生成掩码
            self.predictor.set_image(img)
            point_labels = np.array(labels)
            point_coords = np.stack(np.array(points), axis=0)

            masks, _, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=None,
                multimask_output=True
            )

            return {
                "success": True,
                "detection_count": detected_count,
                "image_path": img_path,
                "error": None,
                "has_mask": masks is not None
            }

        except Exception as e:
            return {
                "success": False,
                "detection_count": 0,
                "image_path": img_path,
                "error": str(e)
            }

    def evaluate_dataset(self, image_dir: str) -> Dict:
        """
        评估数据集，计算检测成功率和统计信息
        """
        results = {
            "total_images": 0,
            "successful_images": 0,  # 成功处理的图片数
            "total_chickens_detected": 0,  # 检测到的总鸡只数
            "failed_images": 0,
            "images_with_detections": 0,  # 有检测到鸡只的图片数
            "detection_details": [],
            "error_details": []
        }

        print(f"\n开始评估文件夹: {image_dir}")
        image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
        # demo_imgs/3_CH251930825_20221102090002.jpg 61
        results["total_images"] = len(image_paths)

        # 检测结果统计
        detection_counts = []  # 记录每张图片检测到的鸡只数量

        for img_path in tqdm(image_paths, desc="处理进度"):
            result = self.process_single_image(str(img_path))

            # 记录检测数量
            detection_count = result["detection_count"]
            detection_counts.append(detection_count)

            if result["success"]:
                results["successful_images"] += 1
                results["total_chickens_detected"] += detection_count
                if detection_count > 0:
                    results["images_with_detections"] += 1

                results["detection_details"].append({
                    "image": str(img_path),
                    "chickens_detected": detection_count,
                    "has_mask": result.get("has_mask", False)
                })
            else:
                results["failed_images"] += 1
                results["error_details"].append({
                    "image": str(img_path),
                    "error": result["error"]
                })

        # 计算统计指标
        total_images = results["total_images"]
        if total_images > 0:
            results.update({
                "success_rate": (results["successful_images"] / total_images) * 100,
                "detection_rate": (results["images_with_detections"] / total_images) * 100
            })

        # 计算检测分布
        detection_distribution = {}
        for count in detection_counts:
            detection_distribution[str(count)] = detection_distribution.get(str(count), 0) + 1
        results["detection_distribution"] = detection_distribution

        return results


def main():
    print("=== 鸡只自动计数系统 ===")

    try:
        print("\n初始化模型...")
        counter = ChickenCounter(model_cfg_path='cfg.ini')

        print("\n开始评估数据集...")
        results = counter.evaluate_dataset("demo_imgs/")

        # 打印详细结果
        print("\n=== 评估结果 ===")
        print(f"总图片数量: {results['total_images']}")
        print(f"成功处理图片数: {results['successful_images']}")
        print(f"处理失败图片数: {results['failed_images']}")
        print(f"\n检测到鸡只的图片数: {results['images_with_detections']}")
        print(f"检测到的总鸡只数: {results['total_chickens_detected']}")
        print(f"\n图片处理成功率: {results['success_rate']:.2f}%")
        print(f"鸡只检出率: {results['detection_rate']:.2f}%")

        print("\n检测数量分布:")
        for count, num in sorted(results["detection_distribution"].items()):
            print(f"检测到 {count} 只鸡的图片数量: {num}")

        # 保存详细结果
        output_file = "鸡只检测评估结果.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\n详细结果已保存到 '{output_file}'")

        # 如果有错误，打印错误详情
        if results["error_details"]:
            print("\n处理失败的图片:")
            for error in results["error_details"]:
                print(f"图片: {error['image']}")
                print(f"错误: {error['error']}")

    except Exception as e:
        print(f"\n系统错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
