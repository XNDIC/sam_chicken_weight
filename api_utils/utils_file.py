import logging
import os
import numpy as np
import cv2
import pdb
from PIL import Image, ImageDraw, ImageFont
import  datetime
import json
import base64
import time
import configparser
import shutil
# 设置日志
logger = logging.getLogger()

def log_setting(log_path):
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    time_rotating_file_handler = logging.handlers.TimedRotatingFileHandler(filename=log_path, when='D')
    time_rotating_file_handler.setLevel(logging.INFO)
    time_rotating_file_handler.setFormatter(formatter)
    logger.addHandler(time_rotating_file_handler)
    return logger

def parse_cfg(cfg_path="cfg.ini"):
    '''
    Args:
        cfg_path (str):
    Returns:
        dict
    '''
    cfg_dict = dict()
    cf = configparser.ConfigParser()
    cf.read(cfg_path)

    # device_info
    cfg_dict["DZ_device_ids_txt"] = cf.get("device_info", "DZ_device_ids_txt")
    cfg_dict["YG_device_ids_txt"] = cf.get("device_info", "YG_device_ids_txt")
    cfg_dict["YG_cheng_ids_txt"] = cf.get("device_info", "YG_cheng_ids_txt")
    cfg_dict["DZ_cheng_ids_txt"] = cf.get("device_info", "DZ_cheng_ids_txt")
    cfg_dict["SZ_cheng_ids_txt"] = cf.get("device_info", "SZ_cheng_ids_txt")
    cfg_dict["SZ_device_ids_txt"] = cf.get("device_info", "SZ_device_ids_txt")
    cfg_dict["HZ_cheng_ids_txt"] = cf.get("device_info", "HZ_cheng_ids_txt")
    cfg_dict["HZ_device_ids_txt"] = cf.get("device_info", "HZ_device_ids_txt")
    cfg_dict["CK_cheng_ids_txt"] = cf.get("device_info", "CK_cheng_ids_txt")
    cfg_dict["CK_device_ids_txt"] = cf.get("device_info", "CK_device_ids_txt")
    cfg_dict["WZ_device_ids_txt"] = cf.get("device_info", "WZ_device_ids_txt")

    # cfg_info
    cfg_dict["log_dir"] = cf.get("cfg_info", "log_dir")
    cfg_dict["check_duration"] = int(cf.get("cfg_info", "check_duration"))
    cfg_dict["device"] = cf.get("cfg_info", "device")

    # model info
    cfg_dict["YG_model_detc"] = cf.get("model_info", "YG_model_detc")
    cfg_dict["DZ_model_detc"] = cf.get("model_info", "DZ_model_detc")
    cfg_dict["SZ_model_detc"] = cf.get("model_info", "SZ_model_detc")
    cfg_dict["LY_model_detc"] = cf.get("model_info", "LY_model_detc")
    cfg_dict["CK_model_detc"] = cf.get("model_info", "CK_model_detc")
    cfg_dict["WZ_model_detc"] = cf.get("model_info", "WZ_model_detc")
    
    # [inference_info]
    cfg_dict["imgsz"] = int(cf.get("inference_info", "imgsz"))
    cfg_dict["conf_thres"] = float(cf.get("inference_info", "conf_thres"))
    cfg_dict["iou_thres"] = float(cf.get("inference_info", "iou_thres"))
    cfg_dict["project"] = cf.get("inference_info", "project")
    cfg_dict["name"] = cf.get("inference_info", "name")

    if not os.path.exists(cfg_dict["log_dir"]):
        os.makedirs(cfg_dict["log_dir"], exist_ok=True)

    return cfg_dict

def mkdir(dir):
    if not os.path.exists(dir):
        logger.info('creating dir: {}'.format(dir))
        os.mkdir(dir)

def mkdirp(dir):
    if not os.path.exists(dir):
        logger.info('creating dir: {}'.format(dir))
        os.makedirs(dir, exist_ok=True)

def txt2list(txt_name):
    txt_list = []
    with open(txt_name) as d:
        all_lines = d.readlines()
        for line in all_lines:
            txt_list.append(line.split("\n")[0])
    print(txt_list)
    return  txt_list


# 时间转为时间戳
def time2stamp(time_cheng):
    split_date = time_cheng.split(' ')[0]
    split_minute = time_cheng.split(' ')[1]
    dd = ""
    for d in split_date.split('-'):
        dd += d
    tt = ""
    for t in split_minute.split(':'):
        tt += t
    cheng_time = dd + tt + "000"
    return cheng_time


# 图片中嵌入鸡体重的文本
def cv2ImgAddText(img, text, left, top, textColor=(255, 255, 255), textSize=50):
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype('fonts/Deng.ttf', textSize,
                                  encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# 图像左上角和右下角分别嵌入设备编号和时间
def insert_time_device(image, device, time_stamp):
    H, W = image.shape[0], image.shape[1]
    jjh = cv2.imread("./bg_img/jjh.png")
    time_block = cv2.imread("./bg_img/time_block.png")
    device_block = cv2.imread("./bg_img/device_block.png")
    jjh = cv2.resize(jjh, (W // 2, W // 2))
    jjh_idx = np.where(jjh > 0)
    jjh_H, jjh_W = jjh.shape[0], jjh.shape[1]
    image[(H // 2 - jjh_H // 2):(H // 2 - jjh_H // 2) + jjh_H, (W // 2 - jjh_W // 2):(W // 2 - jjh_W // 2) + jjh_W][
        jjh_idx] = jjh[jjh_idx]
    tf = time_stamp[:14]
    tt = datetime.datetime.strptime(tf, "%Y%m%d%H%M%S")
    time = str(tt)
    device_block_H, device_block_W = device_block.shape[0], device_block.shape[1]
    time_block_H, time_block_W = time_block.shape[0], time_block.shape[1]
    time_block = cv2ImgAddText(time_block, "时间：{}".format(time), time_block_W // 6, time_block_H // 3)
    device_block = cv2ImgAddText(device_block, "设备编号：{}".format(device), device_block_W // 6, device_block_H // 3)
    H, W = image.shape[0], image.shape[1]
    device_hecheng = cv2.addWeighted(src1=device_block, alpha=0.8,
                                     src2=image[30: device_block_H + 30, 30:device_block_W + 30], beta=0.5, gamma=1)
    time_hecheng = cv2.addWeighted(src1=time_block, alpha=0.8,
                                   src2=image[H - time_block_H - 30: H - 30, W - 30 - time_block_W: W - 30], beta=0.5,
                                   gamma=1)
    image[H - time_block_H - 30: H - 30, W - 30 - time_block_W: W - 30] = time_hecheng
    image[30: device_block_H + 30, 30:device_block_W + 30] = device_hecheng
    # cv2.imwrite("bt_image_1.jpg", image)
    return image

# 判断文本框和原图大小是否匹配
def check_complete(image, center,number_block_H, number_block_W, block_idx):
    tmp_img = image[center[1] - 100:center[1] + number_block_H - 100, center[0] + 100:center[0] + number_block_W + 100]
    tmp_H, tmp_W = tmp_img.shape[0], tmp_img.shape[1]
    for i in block_idx[0]:
        if i > tmp_H:
            return  False
    for j in block_idx[1]:
        if j > tmp_W:
            return False

    return True



# 保存分割的伪标签
def save_pseudo_label(imagePath, result,mode='YG',save_json=True):
    bbox_result, segm_results = result
    shapes_dic_list = []
    save_path = f'./image_save/{mode}'
    mkdir(save_path)
    name = imagePath.split('/')[-1]
    now_time = time.strftime('%Y-%m-%d', time.localtime())
    save_path = os.path.join(save_path, now_time)
    mkdir(save_path)
    image_rgb = cv2.imread(imagePath)
    if not save_json:
        shutil.copy(imagePath,os.path.join(save_path, name))
        return
    h, w ,_ = image_rgb.shape
    for segm_result in segm_results[0]:
        mask = np.uint8(np.ones([h,w])*255)
        mask = mask*segm_result
        ret,thresh = cv2.threshold(mask,1,255,0)
        contours,_ = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for i,contour in enumerate(contours):
            if i ==0 :
                max_point = contour.shape[0]
                best_index = 0
            elif  contour.shape[0] > max_point:
                best_index = i

        points_list = np.squeeze(contours[best_index]).astype(np.int32).tolist()
        shapes_dic = {"label": "chicken","group_id": None,"points": points_list,
                  "shape_type": "polygon","flags": {}}
        shapes_dic_list.append(shapes_dic)

    with open(imagePath, "rb") as f:
        imageData = f.read()
        imageData = base64.b64encode(imageData).decode("utf-8")

    maks_dic = {"version": "3.16.7", "flags": {}, "shapes": shapes_dic_list, "imagePath": imagePath,
                "imageData": imageData,
                "lineColor": [0, 255, 0, 128], "fillColor": [255, 0, 0, 128],
                "imageHeight": h, "imageWidth": w, }

    with open(os.path.join(save_path, name.replace('.jpg', '.json')), "w") as f:
        json.dump(maks_dic, f, ensure_ascii=False, indent=2)

    cv2.imwrite(os.path.join(save_path, name), image_rgb)
    logger.info('generate image pseudo label: {}'.format(os.path.join(save_path, name)))


def insert_weight_classical_result(image, center, weights, classic):
    number_block = cv2.imread("./bg_img/weight_block.png")
    number_block_H, number_block_W = number_block.shape[0], number_block.shape[1]
    # number_block = cv2.resize(number_block,(new_block_H, new_block_W))
    number_block = cv2ImgAddText(number_block, "{}g".format(weights), number_block_W // 3, 30)
    number_block_idx = np.where(number_block > 0)

    classic_block = cv2.imread("./bg_img/classic_block_2.png")
    classic_block_H, classic_block_W = classic_block.shape[0], classic_block.shape[1]
    # new_block_H = classic_block_H // 7
    # ratio = classic_block_W // classic_block_H
    # new_block_W = int(new_block_H * ratio)
    # classic_block = cv2.resize(classic_block,(new_block_H, new_block_W))
    classic_block = cv2ImgAddText(classic_block, "{}".format(classic), classic_block_W // 3, 30)
    classic_block_idx = np.where(classic_block > 0)

    # tmp_img = image[center[1] + 100:center[1] + classic_block_H + 100, center[0] + 100:center[0] + classic_block_W + 100]
    #result = check_complete(tmp_img, center, classic_block_H, classic_block_W, classic_block_idx)
    #if result == False:
        #return image, False
    # if center[1] > image.shape[1]-300 or center[0] > image.shape[0]-300:

    image[center[1] + 100:center[1] + classic_block_H + 100, center[0] + 100:center[0] + classic_block_W + 100][
            classic_block_idx] = classic_block[classic_block_idx]

    # tmp_img = image[center[1] - 100:center[1] + number_block_H - 100, center[0] + 100:center[0] + number_block_W + 100]
    #result = check_complete(tmp_img, center, number_block_H, number_block_W, number_block_idx)
    #if result == False:
        #return image, False
    image[center[1] - 100:center[1] + number_block_H - 100, center[0] + 100:center[0] + number_block_W + 100][
        number_block_idx] = number_block[number_block_idx]
    start_point = (center[0], center[1])
    end_point_1 = (center[0] + 100, center[1] + 150)
    end_point_2 = (center[0] + 100, center[1] - 50)
    cv2.line(image, start_point, end_point_1, (41, 182, 48), 2)
    cv2.line(image, start_point, end_point_2, (41, 182, 48), 2)

    return image, True
