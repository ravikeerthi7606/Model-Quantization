
import torch


def quantize_tensor(tensor: torch.Tensor):
    q_min, q_max = -128, 127

    # Calculate scale
    scale = torch.max(torch.abs(tensor)) / q_max

    # Handle zero tensor
    scale = torch.where(
        scale == 0,
        torch.ones_like(scale),
        scale
    )

    # Quantize
    q_tensor = torch.round(tensor / scale)

    # Clamp values
    q_tensor = torch.clamp(q_tensor, q_min, q_max)

    return q_tensor.to(torch.int8), scale


def dequantize_tensor(
    q_tensor: torch.Tensor,
    scale: torch.Tensor
):
    return q_tensor.to(torch.float32) * scale
