
import torch


def quantize_per_channel(
    tensor: torch.Tensor,
    axis: int = 0
):
    q_min, q_max = -128, 127

    # Normalize axis
    axis = axis % tensor.ndim

    # Dimensions to reduce
    reduce_dims = tuple(
        i for i in range(tensor.ndim)
        if i != axis
    )

    # Calculate maximum absolute value per channel
    max_abs = torch.amax(
        torch.abs(tensor),
        dim=reduce_dims,
        keepdim=True
    )

    # Calculate individual channel scales
    scale = max_abs / q_max

    # Handle zero-valued channels
    scale = torch.where(
        scale == 0,
        torch.ones_like(scale),
        scale
    )

    # Quantization
    q_tensor = torch.round(tensor / scale)

    q_tensor = torch.clamp(
        q_tensor,
        q_min,
        q_max
    )

    return q_tensor.to(torch.int8), scale


def dequantize_per_channel(
    q_tensor: torch.Tensor,
    scale: torch.Tensor
):
    return q_tensor.to(torch.float32) * scale
