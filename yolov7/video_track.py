"""
main code for track
"""
import numpy as np
import torch
import cv2
import pdb
from PIL import Image
import tqdm

import argparse
import os
from time import gmtime, strftime
from tracker.timer import Timer
import yaml
from PIL import Image, ImageDraw, ImageFont

from tracker.basetrack import BaseTracker  # for framework
from tracker.deepsort import DeepSORT
from tracker.bytetrack import ByteTrack
from tracker.deepmot import DeepMOT
from tracker.botsort import BoTSORT
from tracker.uavmot import UAVMOT
from tracker.strongsort import StrongSORT
from yolov7.api_detect import API_Detect



import tracker.tracker_dataloader as tracker_dataloader
import tracker.trackeval as trackeval


def set_basic_params(cfgs):
    global CATEGORY_DICT, DATASET_ROOT, CERTAIN_SEQS, IGNORE_SEQS, YAML_DICT
    CATEGORY_DICT = cfgs['CATEGORY_DICT']
    DATASET_ROOT = cfgs['DATASET_ROOT']
    CERTAIN_SEQS = cfgs['CERTAIN_SEQS']
    IGNORE_SEQS = cfgs['IGNORE_SEQS']
    YAML_DICT = cfgs['YAML_DICT']


timer = Timer()
seq_fps = []  # list to store time used for every seq


def main(opts):
    # set_basic_params(cfgs)  # NOTE: set basic path and seqs params first

    TRACKER_DICT = {
        'sort': BaseTracker,
        'deepsort': DeepSORT,
        'bytetrack': ByteTrack,
        'deepmot': DeepMOT,
        'botsort': BoTSORT,
        'uavmot': UAVMOT,
        'strongsort': StrongSORT,
    }  # dict for trackers, key: str, value: class(BaseTracker)

    # NOTE: ATTENTION: make kalman and tracker compatible
    if opts.tracker == 'botsort':
        opts.kalman_format = 'botsort'
    elif opts.tracker == 'strongsort':
        opts.kalman_format = 'strongsort'

    # NOTE: if save video, you must save image
    if opts.save_videos:
        opts.save_images = True

    """
    1. load model
    """

    model = API_Detect(device='2', weights="model/best_Light_lveyang_V7_v1.1.pt")
    vw = save_videos()

    video_path = "YG2.mp4"
    video = cv2.VideoCapture(video_path)

    # 计步字典
    step  = {}


    tracker = TRACKER_DICT[opts.tracker](opts, frame_rate=30,
                                         gamma=opts.gamma)  # instantiate tracker  TODO: finish init params

    results = []  # store current seq results
    frame_id = 0
    i = 0
    while True:
        i += 1
        ret, frame = video.read()
        
        if ret != True:
            continue
        frame = cv2.resize(frame, (1920,1080))

        timer.tic()  # start timing this img
        img0 = frame
        if not i % opts.detect_per_frame:  # if it's time to detect
        
            out,_ = model.inference_track(frame)  # model forward
            # out = out[0]  # NOTE: for yolo v7

            if len(out.shape) == 3:  # case (bs, num_obj, ...)
                # out = out.squeeze()
                # NOTE: assert batch size == 1
                out = out.squeeze(0)
                img0 = img0.squeeze(0)
            # remove some low conf detections
            out = out[out[:, 4] > 0.1]

            # NOTE: yolo v7 origin out format: [xc, yc, w, h, conf, cls0_conf, cls1_conf, ..., clsn_conf]
            if opts.det_output_format == 'yolo':
                cls_conf, cls_idx = torch.max(out[:, 5:], dim=1)
                # out[:, 4] *= cls_conf  # fuse object and cls conf
                out[:, 5] = cls_idx
                out = out[:, :6]

            current_tracks = tracker.update(out, img0)  # List[class(STracks)]i
        
        else:  # otherwize
            # make the img shape (bs, C, H, W) as (C, H, W)
            if len(img0.shape) == 4:
                img0 = img0.squeeze(0)
            current_tracks = tracker.update_without_detection(None, img0)

      
        # save results
        cur_tlwh, cur_id, cur_cls = [], [], []
        for trk in current_tracks:
            bbox = trk.tlwh
            id = trk.track_id
            cls = trk.cls

            # filter low area bbox
            if bbox[2] * bbox[3] > opts.min_area:
                cur_tlwh.append(bbox)
                cur_id.append(id)
                cur_cls.append(cls)
                # results.append((frame_id + 1, id, bbox, cls))

        results.append((frame_id + 1, cur_id, cur_tlwh, cur_cls))
        timer.toc()  # end timing this image
        if i % 2 == 0:
            frame = plot_img(img0, [cur_tlwh, cur_id, cur_cls], step, is_count=True)
        else:
            frame = plot_img(img0, [cur_tlwh, cur_id, cur_cls], step, is_count=False)
        frame_id += 1
        vw.write(frame)


    seq_fps.append(i / timer.total_time)  # cal fps for current seq
    timer.clear()  # clear for next seq

def save_results(folder_name, seq_name, results, data_type='default'):
    """
    write results to txt file

    results: list  row format: frame id, target id, box coordinate, class(optional)
    to_file: file path(optional)
    data_type: write data format
    """
    assert len(results)
    if not data_type == 'default':
        raise NotImplementedError  # TODO
    return folder_name

# 计算中心点的欧式距离
def eucliDist(A, B):

    A = np.array(A)
    B = np.array(B)
    return np.sqrt(sum(np.power((A-B),2)))

# 计算角度差
def eucliAngel(A, B):

    return abs(A-B)

# 计算向量
def compute_vector(v1, v2):

    vector = (v1[0]-v2[0], v1[1]-v2[1])
    return vector

# 计算向量间的夹角
def dot_product_angle(v1, v2):

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle = np.degrees(arccos)
        return angle
    return 0

def plot_img(img, results, step, is_count=False):
    """
    img: np.ndarray: (H, W, C)
    frame_id: int
    results: [tlwhs, ids, clses]
    save_dir: sr

    plot images with bboxes of a seq
    """


    img_ = np.ascontiguousarray(np.copy(img))

    tlwhs, ids, clses = results[0], results[1], results[2]
    
    for tlwh, id, cls in zip(tlwhs, ids, clses):
        # convert tlwh to tlbr
        tlbr = tuple([int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])])
        bottom_left = (tlwh[0], tlwh[3])
        center = (int(tlwh[0] + tlwh[0] + tlwh[2]) // 2, int(tlwh[1] + tlwh[1] + tlwh[3])//2)
        if center[0] > 960 and center[0] < 1000 and center[1] < 300:
            continue 
        if len(step) == 0 or id not in step:

            step.update({id:{"center":center, "step": 0, "angel": 0}})

        last_center = step[id]["center"]
        last_angel = step[id]["angel"]

        vector_1 = compute_vector(center, bottom_left)
        vector_fa = compute_vector(center, (img.shape[1], img.shape[0]))

        angel = dot_product_angle(vector_1, vector_fa)
        print(eucliAngel(angel, last_angel))
        if eucliDist(center, last_center) > 20 or eucliAngel(angel, last_angel) > 5:

            step[id]["step"] += 1

        if is_count:
            step[id]["center"] = center
            step[id]["angel"] = angel

        # draw a rect
        if step[id]["step"] > 1:

            cv2.rectangle(img_, tlbr[:2], tlbr[2:], get_color(id), thickness=3, )

            # note the id and cls
            text = '步数: {}'.format(step[id]["step"])
            img_pil = Image.fromarray(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))

            font = ImageFont.truetype('STHUPO.TTF', 40, encoding="utf-8")
            draw = ImageDraw.Draw(img_pil)
            color = (get_color(id)[2], get_color(id)[1],get_color(id)[0] )
            draw.text((tlbr[0], tlbr[1]-40), text, font=font, fill=(0,255,0))
            img_ = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # cv2.putText(img_, text, (tlbr[0], tlbr[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3,
            #             color=get_color(id), thickness=4)

    return img_


def save_videos(size=(1920,1080)):


    fourcc = cv2.VideoWriter_fourcc(*"mp4v")


    vw = cv2.VideoWriter('./LY_step_counting.avi',cv2.VideoWriter_fourcc("M","J","P","G"),20,size)



    return vw

print('Save videos Done!!')


def get_color(idx):
    """
    aux func for plot_seq
    get a unique color for each id
    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


if __name__ == '__main__':
    # # python tracker/track.py --dataset visdrone --data_format origin --tracker sort --model_path runs/train/yolov7-w6-custom4/weights/best.pt
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='visdrone', help='visdrone, or mot')
    parser.add_argument('--data_format', type=str, default='origin', help='format of reading dataset')
    parser.add_argument('--det_output_format', type=str, default='yolo',
                        help='data format of output of detector, yolo or other')

    parser.add_argument('--tracker', type=str, default='sort', help='sort, deepsort, etc')

    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--trace', type=bool, default=False, help='traced model of YOLO v7')

    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')

    """For tracker"""
    # model path
    parser.add_argument('--reid_model_path', type=str, default='./weights/ckpt.t7', help='path for reid model path')
    parser.add_argument('--dhn_path', type=str, default='./weights/DHN.pth', help='path of DHN path for DeepMOT')

    # threshs
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='filter tracks')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='thresh for NMS')
    parser.add_argument('--iou_thresh', type=float, default=0.8, help='IOU thresh to filter tracks')

    # other options
    parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--gamma', type=float, default=0.1, help='param to control fusing motion and apperance dist')
    parser.add_argument('--kalman_format', type=str, default='default',
                        help='use what kind of Kalman, default, naive, strongsort or bot-sort like')
    parser.add_argument('--min_area', type=float, default=150, help='use to filter small bboxs')

    parser.add_argument('--save_images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save_videos', action='store_true', help='save tracking results (video)')

    # detect per several frames
    parser.add_argument('--detect_per_frame', type=int, default=1, help='choose how many frames per detect')

    parser.add_argument('--track_eval', type=bool, default=True, help='Use TrackEval to evaluate')

    opts = parser.parse_args()
    #
    # # NOTE: read path of datasets, sequences and TrackEval configs
    # with open(f'./tracker/config_files/{opts.dataset}.yaml', 'r') as f:
    #     cfgs = yaml.load(f, Loader=yaml.FullLoader)
    main(opts)
