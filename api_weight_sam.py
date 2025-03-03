# -*- coding: utf-8 -*-
# time: 2024/4/22 15:03
# file: api_weight_sam_new.py
# author: kangzhe.ma
# description : v1.0: 使用sam和yolov7的目标检测进行分割估重展示（4.22）

import cv2
import math
import random
import os
import numpy as np
import time
import datetime

from api_utils.utils_file import insert_time_device, insert_weight_classical_result, txt2list, parse_cfg

import shutil
from logger import init_logger
import logging
from effcientvit_sam import run_point, init_sam
from yolov7.api_detect import API_Detect

init_logger()
cfg = parse_cfg('cfg.ini')
logger = logging.getLogger('chicken_weight')


def init_detc_model(cfg):
    device_id = str(cfg['device']).split(':')[-1]
    YG_model = API_Detect(device=device_id, weights=cfg['YG_model_detc'])
    DZ_model = API_Detect(device=device_id, weights=cfg['DZ_model_detc'])
    SZ_model = API_Detect(device=device_id, weights=cfg['SZ_model_detc'])
    HZ_model = API_Detect(device=device_id, weights=cfg['LY_model_detc'])
    CK_model = API_Detect(device=device_id, weights=cfg['CK_model_detc'])
    WZ_model = API_Detect(device=device_id, weights=cfg['WZ_model_detc'])
    logger.info('load model success!')
    return YG_model, DZ_model, SZ_model, HZ_model, CK_model, WZ_model


def init_device(cfg):
    YG_device_ids = txt2list(cfg['YG_device_ids_txt'])
    DZ_device_ids = txt2list(cfg['DZ_device_ids_txt'])
    SZ_device_ids = txt2list(cfg['SZ_device_ids_txt'])
    HZ_device_ids = txt2list(cfg['HZ_device_ids_txt'])
    CK_device_ids = txt2list(cfg['CK_device_ids_txt'])
    WZ_device_ids = txt2list(cfg['WZ_device_ids_txt'])
    logger.info('load config devices success!')
    return YG_device_ids, DZ_device_ids, SZ_device_ids, HZ_device_ids, CK_device_ids, WZ_device_ids


# 功能函数：当前路径的文件复制到目标路径
def uploadfile(remotePath, localPath):
    """
    :param remotePath:
    :param localPath:
    :return:
    """
    print("start copy file by use SFTP...")
    logger.info("start upload file by use SFTP...")
    result = [1, ""]
    try:
        shutil.move(localPath, remotePath)
        result = [1, "upload " + remotePath + " success"]
    except Exception as e:
        result = [-1, "upload fail, reason:{0}".format(e)]
    print(result[1])
    return result


def mask2polygon(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #
    # print(len(contours))
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # print(largest_contour)
    return largest_contour


def detc_chicken(model, img_path):
    # 返回point,bbox
    _, number, bbox_list = model.inference(img_path)

    return bbox_list


def judge_spot(device_id, model_info_dict):
    spot = None
    for key, value in model_info_dict.items():
        if device_id in value[1]:
            spot = key
    if spot is None:
        logger.info(f'{device_id} is not support,please check device list!')
    return spot


def bbox2sam_points(bbox_list, img_shape):
    points, labels, best_c = [], [], []
    H, W, _ = img_shape
    cx, cy = W // 2, H // 2
    max_distancs = float('inf')

    for bbox in bbox_list:
        x1, y1, x2, y2 = bbox
        x_c, y_c = int((x1 + x2) / 2), int((y1 + y2) / 2)
        #  保证在圆环内
        distance = math.sqrt((x_c - cx) ** 2 + (y_c - cy) ** 2)
        if distance > H / 3:
            continue
        # 选最近的鸡
        if distance < max_distancs:
            max_distancs = distance
            points = [(x_c, y_c), (x1, x2), (y1, y2)]
            labels = [1, 0, 0]
            best_c = [x_c, y_c]
            # points = [(x_c,y_c)]
            # labels = [1]
    return points, labels, best_c


def inference(remotepath, img_name, img_time, device_id, species, weight_range, detc_model, sam_modle):
    img_path = remotepath + '/' + img_name
    _, number, bbox_list = detc_model.inference(img_path)
    img = cv2.imread(img_path)
    os.remove(img_path)

    # todo 这里加回归的代码
    weight = random.randint(weight_range[0], weight_range[1])

    points, labels, bbox_c = bbox2sam_points(bbox_list, img.shape)
    # 判断灰度图
    a = img[:, :, 0] == img[:, :, 1]
    if len(points) == 0 or len(labels) == 0 or len(bbox_c) == 0 or a.all() == True:
        logger.info(f'{img_path} have no chicken or dark img!')
        return img, weight, False

    chicken_mask = run_point(sam_modle, img, points, labels)
    chicken_mask = np.array(chicken_mask).astype(np.uint8)

    # 判断mask是否面积过大
    non_zero_pixels = cv2.countNonZero(chicken_mask)
    non_zero_ratio = non_zero_pixels / (img.shape[0] * img.shape[1])
    # print(non_zero_ratio)
    if non_zero_ratio > 0.01:
        return img, weight, False

    chicken_contour = mask2polygon(chicken_mask)
    img = insert_time_device(img, device_id, str(img_time))
    cv2.drawContours(img, chicken_contour, -1, (0, 255, 0), 3)

    img, _ = insert_weight_classical_result(img, bbox_c, weight, species)

    return img, weight, True


def main_pipline(remotepath):
    # step1: 初始化模型和配置
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    YG_model, DZ_model, SZ_model, HZ_model, CK_model, WZ_model = init_detc_model(cfg)
    sam_modle, _, _ = init_sam('cuda:0')
    YG_device_ids, DZ_device_ids, SZ_device_ids, HZ_device_ids, CK_device_ids, WZ_device_ids = init_device(cfg)
    model_info_dict = {'YG': [YG_model, YG_device_ids, '黄鸡', (1500, 1600)],
                       'DZ': [DZ_model, DZ_device_ids, '芦花鸡', (1500, 1600)],
                       'SZ': [SZ_model, SZ_device_ids, '黄鸡', (1000, 1150)],
                       'HZ': [HZ_model, HZ_device_ids, '乌鸡', (1500, 1600)],
                       'CK': [CK_model, CK_device_ids, '乌鸡', (1500, 1600)],
                       'WZ': [WZ_model, WZ_device_ids, '土鸡', (1000, 1300)]}

    # step2:遍历图片，判断图片来自于哪个设备，调用哪个模型
    rightnow_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_time = str(datetime.datetime.now()).split(" ")[0]
    upload_path = '/jiezhi.yang/chicken_files_new/put/pub/2-2/'

    while True:
        try:
            img_list = os.listdir(remotepath)
            for img_name in img_list:
                img_time = img_name.split(".")[0].split("_")[-1][:14]
                device_id = img_name.split("_")[1]
                spot = judge_spot(device_id, model_info_dict)
                species = model_info_dict[spot][2]
                weight_range = model_info_dict[spot][3]
                detc_model = model_info_dict[spot][0]
                img, weight, success = inference(remotepath, img_name, img_time, device_id, species, weight_range,
                                                 detc_model, sam_modle)
                if success:
                    img = cv2.resize(img, (960, 540))
                    cv2.imwrite('./tmp/image-' + str(weight) + '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    uploadfile(upload_path + img_name.split('.jpg')[0] + '_' + str(weight) + '.jpg',
                               './tmp/image-' + str(weight) + '.jpg')
                    logger.info(f'[{spot}] : inferecn success! {img_name}')
            logger.info(f'The service is waiting!')
            time.sleep(cfg['check_duration'])
        except Exception as e:
            logger.info(f'The service is waiting! because the error {e}')
            time.sleep(cfg['check_duration'])


if __name__ == "__main__":
    main_pipline('/jiezhi.yang/chicken_files_new/put/pub/2')
