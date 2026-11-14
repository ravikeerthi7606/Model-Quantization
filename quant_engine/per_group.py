
import torch
import torch.nn.functional as F


def quantize_per_group(
    tensor: torch.Tensor,
    group_size: int = 4,
    axis: int = -1
):
    if tensor.ndim == 0 or tensor.numel() == 0:
        raise ValueError("Tensor must be non-empty and non-scalar")

    if group_size <= 0:
        raise ValueError("group_size must be positive")

    if not tensor.is_floating_point():
        raise TypeError("Input tensor must be floating point")

    q_min, q_max = -128, 127

    axis = axis % tensor.ndim

    # Move selected axis to the last dimension
    x = tensor.movedim(axis, -1)

    original_size = x.shape[-1]

    # Calculate padding
    padding = (group_size - original_size % group_size) % group_size

    if padding:
        x = F.pad(x, (0, padding))

    # Divide into groups
    grouped = x.reshape(
        *x.shape[:-1],
        -1,
        group_size
    )

    # Calculate scale for each group
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

    # Quantize
    q = torch.round(grouped / scale)
    q = torch.clamp(q, q_min, q_max)

    # Restore original shape
    q = q.reshape(*x.shape)
    q = q[..., :original_size]
    q = q.movedim(-1, axis)

    return q.to(torch.int8), scale


def dequantize_per_group(
    q_tensor: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 4,
    axis: int = -1
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

    reconstructed = grouped.to(torch.float32) * scale

    reconstructed = reconstructed.reshape(*q.shape)
    reconstructed = reconstructed[..., :original_size]
    reconstructed = reconstructed.movedim(-1, axis)

    return reconstructed
