
import torch
import torch.nn as nn

from .utils import get_quantization_range
from .observers import ModelCalibrator
from .model_conversion import quantize_model


class ActivationFakeQuantizer(nn.Module):

    def __init__(self, scale, bits=8):

        super().__init__()

        q_min, q_max = get_quantization_range(bits)

        scale = torch.as_tensor(
            scale,
            dtype=torch.float32
        ).detach().clone()

        if scale.numel() != 1:
            raise ValueError("Activation scale must be scalar")

        if not torch.isfinite(scale).all() or scale.item() <= 0:
            raise ValueError("Activation scale must be finite and positive")

        self.register_buffer("scale", scale)

        self.q_min = q_min
        self.q_max = q_max
        self.bits = bits

    def forward(self, x):

        scale = self.scale.to(
            device=x.device,
            dtype=x.dtype
        )

        q = torch.round(x / scale)

        q = torch.clamp(
            q,
            self.q_min,
            self.q_max
        )

        return q * scale


def insert_activation_quantizers(
    model,
    calibration_results,
    bits=8
):

    def replace_layers(parent, prefix=""):

        for name, child in list(parent.named_children()):

            full_name = (
                f"{prefix}.{name}"
                if prefix
                else name
            )

            if full_name in calibration_results:

                scale = calibration_results[full_name]["scale"]

                wrapper = nn.Sequential(
                    child,
                    ActivationFakeQuantizer(
                        scale=scale,
                        bits=bits
                    )
                )

                setattr(parent, name, wrapper)

            else:

                replace_layers(
                    child,
                    full_name
                )

    replace_layers(model)

    return model


def ptq_quantize_model(
    model,
    calibration_data,
    bits=8,
    granularity="per_channel",
    group_size=128,
    axis=0,
    calibration_method="minmax",
    percentile=99.9
):

    # Step 1: Collect activation statistics
    calibrator = ModelCalibrator(
    bits=bits,
    observer_type=calibration_method,
    percentile=percentile
)

    calibrator.calibrate(
        model,
        calibration_data
    )

    calibration_results = (
        calibrator.get_calibration_results()
    )

    # Step 2: Quantize model weights
    quantized_model = quantize_model(
        model,
        bits=bits,
        granularity=granularity,
        group_size=group_size,
        axis=axis
    )

    # Step 3: Insert activation fake quantizers
    quantized_model = insert_activation_quantizers(
        quantized_model,
        calibration_results,
        bits=bits
    )

    return quantized_model, calibration_results
