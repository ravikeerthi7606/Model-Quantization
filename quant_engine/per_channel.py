
import torch
from .utils import get_quantization_range


def quantize_per_channel(tensor, axis=0, bits=8):

    if tensor.ndim == 0 or tensor.numel() == 0:
        raise ValueError("Tensor must be non-empty and non-scalar")

    if not tensor.is_floating_point():
        raise TypeError("Input must be floating point")

    if not torch.isfinite(tensor).all():
        raise ValueError("Tensor contains NaN or Inf")

    if not -tensor.ndim <= axis < tensor.ndim:
        raise ValueError("Invalid axis")

    q_min, q_max = get_quantization_range(bits)

    axis = axis % tensor.ndim

    reduce_dims = tuple(
        i for i in range(tensor.ndim)
        if i != axis
    )

    max_abs = tensor.abs().amax(
        dim=reduce_dims,
        keepdim=True
    )

    scale = max_abs / q_max

    scale = torch.where(
        scale == 0,
        torch.ones_like(scale),
        scale
    )

    q = torch.round(tensor / scale)
    q = torch.clamp(q, q_min, q_max)

    return q.to(torch.int8), scale


def dequantize_per_channel(q_tensor, scale):
    return q_tensor.to(scale.dtype) * scale
