
import math
import torch


def _validate_tensors(original, reconstructed):

    if not isinstance(original, torch.Tensor):
        raise TypeError("original must be a torch.Tensor")

    if not isinstance(reconstructed, torch.Tensor):
        raise TypeError("reconstructed must be a torch.Tensor")

    if original.shape != reconstructed.shape:
        raise ValueError("Tensor shapes must match")

    if not original.is_floating_point():
        raise TypeError("original must be floating point")

    if not reconstructed.is_floating_point():
        raise TypeError("reconstructed must be floating point")

    if original.device != reconstructed.device:
        raise ValueError("Tensors must be on the same device")

    if original.numel() == 0:
        raise ValueError("Tensors cannot be empty")

    if not torch.isfinite(original).all():
        raise ValueError("original contains NaN or Inf")

    if not torch.isfinite(reconstructed).all():
        raise ValueError("reconstructed contains NaN or Inf")


def compute_metrics(original, reconstructed):

    _validate_tensors(original, reconstructed)

    # Use float64 for stable metric calculations
    x = original.to(torch.float64)
    x_hat = reconstructed.to(torch.float64)

    error = x - x_hat

    mse = torch.mean(error ** 2)
    mae = torch.mean(torch.abs(error))
    max_error = torch.max(torch.abs(error))

    signal_power = torch.mean(x ** 2)
    noise_power = mse

    if noise_power.item() == 0:
        sqnr = float("inf")

    elif signal_power.item() == 0:
        sqnr = float("-inf")

    else:
        sqnr = (
            10 * torch.log10(signal_power / noise_power)
        ).item()

    return {
        "mse": mse.item(),
        "mae": mae.item(),
        "max_abs_error": max_error.item(),
        "sqnr_db": sqnr
    }
