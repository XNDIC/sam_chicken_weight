# -*- coding: utf-8 -*-
# time: 2024/4/19 15:23
# file: api_sam_weights.py
# author: kangzhe.ma
# description : sam模型根据点进行分割
import sys
import numpy as np
sys.path.append('/jiezhi.yang/chicken_new/sam_chicken_weight/efficientvit_master')
sys.path.append('/jiezhi.yang/chicken_new/sam_chicken_weight/efficientvit_master/segment_anything')
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator



def init_sam(device):
    efficientvit_sam = create_sam_model(name="xl1", weight_url="efficientvit_master/assets/checkpoints/sam/xl1.pt")
    efficientvit_sam = efficientvit_sam.cuda(device).eval()
    efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)
    efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(efficientvit_sam)
    return  efficientvit_sam,efficientvit_sam_predictor,efficientvit_mask_generator

def predict_mask_from_point(
        predictor: EfficientViTSamPredictor, point_coords: np.ndarray, point_labels: np.ndarray
) -> np.ndarray:
    masks, iou_predictions, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=None,
        multimask_output=True,
    )

    mask = masks[iou_predictions.argmax()]
    return mask

def run_point(efficientvit_sam, image_np, points, labels):
    predictor = EfficientViTSamPredictor(efficientvit_sam)
    predictor.set_image(image_np)

    point_labels = np.array(labels)
    point_coords = np.stack(np.array(points), axis=0)

    pre_mask = predict_mask_from_point(predictor, point_coords, point_labels)
    return pre_mask

