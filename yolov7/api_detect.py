import argparse
import time
from pathlib import Path

# import cv2
import torch
import sys
import os
# import torch.backends.cudnn as cudnn
# from numpy import random

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov7/ to path
# 添加 yolov7 目录到 Python 路径
yolov7_path = os.path.dirname(os.path.abspath(__file__))
if yolov7_path not in sys.path:
    sys.path.append(yolov7_path)
from yolov7.models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from api_utils.utils_file import parse_cfg
from tracker.tracker_dataloader import TrackerLoader


class API_Detect():
    def __init__(self,
                 weights,
                 device,
                 view_img=False,
                 save_txt=False,
                 save_conf=False,
                 nosave=True,
                 classes=None,
                 agnostic_nms=False,
                 augment=False,
                 update=False,
                 exist_ok=False,
                 no_trace=False,
                 half=False,
                 classify=False,
                 ):
        cfg = parse_cfg('cfg.ini')
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.update = update
        self.exist_ok = exist_ok
        self.no_trace = no_trace
        self.half = half
        self.classify = classify
        self.conf_thres, self.iou_thres, self.project, self.name, self.imgsz = cfg['conf_thres'], cfg['iou_thres'], cfg[
            'project'], cfg['name'], \
                                                                               cfg['imgsz']

        self.device = select_device(device)
        print(self.device)
        # Load model
        model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(self.imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        if not no_trace:
            model = TracedModel(model, self.device, imgsz)
        self.stride = stride
        self.model = model
        self.TL = TrackerLoader(self.imgsz, self.stride)

    def inference_track(self, frame):
        img, img0 = self.TL.get_image(frame)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        out = self.model(img, augment=self.augment)[0]  # model forward
        out = non_max_suppression(out, self.conf_thres, self.iou_thres, classes=self.classes,
                                  agnostic=self.agnostic_nms)

        out = out[0]  # NOTE: for yolo v7
        out = xyxy2xywh(out)

        if len(out.shape) == 3:  # case (bs, num_obj, ...)
            # out = out.squeeze()
            # NOTE: assert batch size == 1
            out = out.squeeze(0)
            img0 = img0.squeeze(0)
        # remove some low conf detections
        out = out[out[:, 4] > 0.001]

        # NOTE: yolo v7 origin out format: [xc, yc, w, h, conf, cls0_conf, cls1_conf, ..., clsn_conf]

        cls_conf, cls_idx = torch.max(out[:, 5:], dim=1)
        # out[:, 4] *= cls_conf  # fuse object and cls conf
        out[:, 5] = cls_idx
        out = out[:, :6]
        return out, img0

    def inference(self, source):

        # Set Dataloader
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Warmup
            if self.device.type != 'cpu' and (
                    old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=self.augment)[0]

            # Inference
            pred = self.model(img, augment=self.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes,
                                       agnostic=self.agnostic_nms)

            # Process detections
            bbox_list = []
            chicken_num = 0
            for i, det in enumerate(pred):  # detections per image

                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                if len(det):

                    chicken_num = 0
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        if conf < 0.2:
                            continue
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        bbox_list.append([x1, y1, x2, y2])
                        chicken_num += 1

        print(f'Done. ({time.time() - t0:.3f}s)')
        return im0, chicken_num, bbox_list
