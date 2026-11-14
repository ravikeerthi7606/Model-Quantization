
import torch
import torch.nn.functional as F

from .utils import get_quantization_range


def quantize_per_group(
    tensor,
    group_size=4,
    axis=-1,
    bits=8
):

    if tensor.ndim == 0 or tensor.numel() == 0:
        raise ValueError("Tensor must be non-empty and non-scalar")

    if not tensor.is_floating_point():
        raise TypeError("Input must be floating point")

    if not torch.isfinite(tensor).all():
        raise ValueError("Tensor contains NaN or Inf")

    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")

    if not -tensor.ndim <= axis < tensor.ndim:
        raise ValueError("Invalid axis")

    q_min, q_max = get_quantization_range(bits)

    axis = axis % tensor.ndim

    x = tensor.movedim(axis, -1)

    original_size = x.shape[-1]

    padding = (group_size - original_size % group_size) % group_size

    if padding:
        x = F.pad(x, (0, padding))

    grouped = x.reshape(
        *x.shape[:-1],
        -1,
        group_size
    )

    max_abs = grouped.abs().amax(
        dim=-1,
        keepdim=True
    )

    scale = max_abs / q_max

    scale = torch.where(
        scale == 0,
        torch.ones_like(scale),
        scale
    )

    q = torch.round(grouped / scale)
    q = torch.clamp(q, q_min, q_max)

    q = q.reshape(*x.shape)
    q = q[..., :original_size]
    q = q.movedim(-1, axis)

    return q.to(torch.int8), scale


def dequantize_per_group(
    q_tensor,
    scale,
    group_size=4,
    axis=-1
):

    axis = axis % q_tensor.ndim

    q = q_tensor.movedim(axis, -1)

    original_size = q.shape[-1]

    padding = (group_size - original_size % group_size) % group_size

    if padding:
        q = F.pad(q, (0, padding))

    grouped = q.reshape(
        *q.shape[:-1],
        -1,
        group_size
    )

    reconstructed = grouped.to(scale.dtype) * scale

    reconstructed = reconstructed.reshape(*q.shape)
    reconstructed = reconstructed[..., :original_size]
    reconstructed = reconstructed.movedim(-1, axis)

    return reconstructed
