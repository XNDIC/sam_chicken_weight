# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023
import functools
import os
from inspect import signature
from typing import List, Dict, Any, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "is_parallel",
    "get_device",
    "get_same_padding",
    "resize",
    "build_kwargs_from_config",
    "load_state_dict_from_file",
]


def is_parallel(model: nn.Module) -> bool:
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def get_device(model: nn.Module) -> torch.device:
    return model.parameters().__next__().device


def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def resize(
    x: torch.Tensor,
    size: Optional[Any] = None,
    scale_factor: Optional[List[float]] = None,
    mode: str = "bicubic",
    align_corners: Optional[bool] = False,
    # x: torch.Tensor,
    # size: any or None = None,
    # scale_factor: list[float] or None = None,
    # mode: str = "bicubic",
    # align_corners: bool or None = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


# def build_kwargs_from_config(config: dict, target_func: callable) -> dict[str, any]:
# def build_kwargs_from_config(config: Dict[str, Any], target_func: Callable) -> Dict[str, Any]:
#     valid_keys = list(signature(target_func).parameters)
#     kwargs = {}
#     for key in config:
#         if key in valid_keys:
#             kwargs[key] = config[key]
#     return kwargs
def build_kwargs_from_config(config: Dict[str, Any], target_func: Callable) -> Dict[str, Any]:
    """
    从配置中构建参数字典
    """
    try:
        # 处理 partial 对象
        if isinstance(target_func, functools.partial):
            base_func = target_func.func
            # 合并原始参数和配置
            kwargs = dict(target_func.keywords)
            kwargs.update(config)
            return kwargs

        # 处理普通函数或类
        valid_keys = list(signature(target_func).parameters.keys())
        kwargs = {}
        for key in config:
            if key in valid_keys:
                kwargs[key] = config[key]
        return kwargs
    except Exception as e:
        print(f"构建参数时出错: {str(e)}")
        return {}

# def load_state_dict_from_file(file: str, only_state_dict=True) -> dict[str, torch.Tensor]:
def load_state_dict_from_file(file: str, only_state_dict=True) -> Dict[str, torch.Tensor]:
    file = os.path.realpath(os.path.expanduser(file))
    checkpoint = torch.load(file, map_location="cpu")
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint
