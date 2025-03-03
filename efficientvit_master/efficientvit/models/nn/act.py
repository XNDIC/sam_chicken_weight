# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

from functools import partial

import torch.nn as nn

from typing import List, Dict, Any, Callable, Optional, Union
from efficientvit_master.efficientvit.models.utils import build_kwargs_from_config

__all__ = ["build_act"]


# register activation function here
REGISTERED_ACT_DICT: Dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}


# def build_act(name: str, **kwargs) -> nn.Module or None:
#     if name in REGISTERED_ACT_DICT:
#         act_cls = REGISTERED_ACT_DICT[name]
#         args = build_kwargs_from_config(kwargs, act_cls)
#         return act_cls(**args)
#     else:
#         return None
def build_act(act_func: Optional[str], **kwargs) -> Optional[nn.Module]:
    """
    构建激活函数
    """
    if act_func is None:
        return None

    if act_func == "gelu":
        return nn.GELU()  # 不使用 tanh 近似
    elif act_func == "relu":
        return nn.ReLU()
    elif act_func == "relu6":
        return nn.ReLU6()
    elif act_func == "hswish":
        return nn.Hardswish()
    else:
        raise ValueError(f"不支持的激活函数: {act_func}")
