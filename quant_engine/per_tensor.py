
import torch

from .utils import get_quantization_range


def quantize_tensor(tensor, bits=8):

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    if tensor.ndim == 0 or tensor.numel() == 0:
        raise ValueError("Tensor must be non-empty and non-scalar")

    if not tensor.is_floating_point():
        raise TypeError("Input must be floating point")

    if not torch.isfinite(tensor).all():
        raise ValueError("Tensor contains NaN or Inf")

    q_min, q_max = get_quantization_range(bits)

    max_abs = torch.max(torch.abs(tensor))

    scale = max_abs / q_max

    scale = torch.where(
        scale == 0,
        torch.ones_like(scale),
        scale
    )

    q_tensor = torch.round(tensor / scale)

    q_tensor = torch.clamp(
        q_tensor,
        q_min,
        q_max
    )

    return q_tensor.to(torch.int8), scale


def dequantize_tensor(q_tensor, scale):
    return q_tensor.to(torch.float32) * scale
