import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from yolov7.api_detect import API_Detect
from api_utils.utils_file import parse_cfg
import random

# 加载YOLOv7模型配置
cfg = parse_cfg('cfg.ini')
DZ_model = API_Detect(device='0', weights=cfg['DZ_model_detc'])


# 图像处理与检测边界框
def load_image_and_detect_bbox(img_path):
    _, _, bbox_list = DZ_model.inference(img_path)
    return bbox_list


# 特征提取函数：从边界框提取特征
def bbox_to_features(bbox_list):
    features = []
    for bbox in bbox_list:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        features.append([width, height, area])
    return np.array(features)


# 数据准备
def prepare_dataset(image_paths):
    all_features = []
    all_weights = []
    for img_path in image_paths:
        weights = []
        bbox_list = load_image_and_detect_bbox(img_path)
        features = bbox_to_features(bbox_list)
        if features.shape[0] == 0:  # 无检测到的边界框，跳过
            continue
        else:
            for i in range(len(features)):
                weights.append(round(random.uniform(1, 3), 2))
        all_features.extend(features)
        all_weights.extend(weights)
        # print("all_features", all_features)
        # print("all_weights", all_weights)
    return np.array(all_features), np.array(all_weights)


# 模型训练
def train_model(features, weights):
    X_train, X_test, y_train, y_test = train_test_split(features, weights, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse:.2f}')

    return model


# 预测
def predict_weight(model, image_path):
    bbox_list = load_image_and_detect_bbox(image_path)
    features = bbox_to_features(bbox_list)
    predicted_weights = model.predict(features)
    return predicted_weights


# 图片路径作为训练数据
image_paths = ['demo_imgs/T_K88061537_20221210135901.jpg']
# actual_weights = [1.5, 2.0, 2.5]

# 准备数据集
features, weights = prepare_dataset(image_paths)

# 训练模型
model = train_model(features, weights)

# 进行预测
test_image = 'demo_imgs/3_CH251930825_20221102090002.jpg'  # 待检测的图片路径
predicted_weights = predict_weight(model, test_image)

# 打印预测结果
print(f'Predicted weights for the chickens in the image: {predicted_weights}')
