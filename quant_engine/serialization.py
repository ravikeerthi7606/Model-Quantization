from pathlib import Path
import math
import torch

from .packing import pack_int4, unpack_int4
from .quantizer import Quantizer


FORMAT_VERSION = 1

_DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.bfloat16": torch.bfloat16,
}


def _validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")

    required_fields = [
        "bits",
        "scheme",
        "granularity",
        "original_shape",
        "original_numel",
        "original_dtype",
        "packing",
    ]

    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Missing metadata field: {field}")

    shape = tuple(metadata["original_shape"])
    if not shape or any(
        not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
        for dim in shape
    ):
        raise ValueError("original_shape must contain non-negative integers")

    numel = math.prod(shape)
    if numel != metadata["original_numel"]:
        raise ValueError("original_numel does not match original_shape")

    if metadata["bits"] not in (4, 8):
        raise ValueError("bits must be 4 or 8")

    if metadata["scheme"] != "symmetric":
        raise ValueError("Unsupported quantization scheme")

    if metadata["granularity"] not in (
        "per_tensor", "per_channel", "per_group"
    ):
        raise ValueError("Invalid granularity")

    if metadata["original_dtype"] not in _DTYPE_MAP:
        raise ValueError("Unsupported original dtype")

    if metadata["granularity"] in ("per_channel", "per_group"):
        if not isinstance(metadata.get("axis"), int):
            raise ValueError("axis is required for per-channel/per-group")

    if metadata["granularity"] == "per_group":
        group_size = metadata.get("group_size")
        if (
            not isinstance(group_size, int)
            or isinstance(group_size, bool)
            or group_size <= 0
        ):
            raise ValueError("A positive group_size is required")


def save_quantized(payload, scales, metadata, path):
    """Save a quantized payload, scale tensor, and self-describing metadata."""
    if not isinstance(payload, torch.Tensor):
        raise TypeError("payload must be a torch.Tensor")

    if not isinstance(scales, torch.Tensor):
        scales = torch.as_tensor(scales)

    _validate_metadata(metadata)

    if not scales.is_floating_point() or not torch.isfinite(scales).all():
        raise ValueError("scales must contain finite floating-point values")

    artifact = {
        "format_version": FORMAT_VERSION,
        "payload": payload.detach().cpu(),
        "scales": scales.detach().cpu(),
        "metadata": dict(metadata),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


def load_quantized(path, map_location="cpu"):
    """Load and validate a previously saved quantized artifact."""
    artifact = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )

    if not isinstance(artifact, dict):
        raise ValueError("Invalid quantized artifact")

    if artifact.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported serialization format version")

    for field in ("payload", "scales", "metadata"):
        if field not in artifact:
            raise ValueError(f"Artifact is missing {field}")

    _validate_metadata(artifact["metadata"])

    return (
        artifact["payload"],
        artifact["scales"],
        artifact["metadata"],
    )


def quantize_and_save(tensor, quantizer, path):
    """
    Quantize a tensor and save a self-contained artifact.

    INT4 values are nibble-packed. INT8 values are stored directly.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")

    q_tensor, params = quantizer.quantize(tensor)

    packing_metadata = None
    if quantizer.bits == 4:
        payload, packing_metadata = pack_int4(q_tensor)
    else:
        payload = q_tensor

    metadata = {
        "bits": quantizer.bits,
        "scheme": quantizer.scheme,
        "granularity": quantizer.granularity,
        "original_shape": tuple(tensor.shape),
        "original_numel": tensor.numel(),
        "original_dtype": str(tensor.dtype),
        "axis": params.get("axis"),
        "group_size": params.get("group_size"),
        "packing": packing_metadata,
    }

    save_quantized(payload, params["scale"], metadata, path)
    return metadata


def load_and_dequantize(path, map_location="cpu"):
    """
    Load a quantized artifact and reconstruct its floating-point tensor.
    No separately configured Quantizer is required.
    """
    payload, scales, metadata = load_quantized(
        path,
        map_location=map_location,
    )

    bits = metadata["bits"]

    if bits == 4:
        if metadata["packing"] is None:
            raise ValueError("INT4 artifact is missing packing metadata")
        q_tensor = unpack_int4(payload, metadata["packing"])
    else:
        q_tensor = payload

    quantizer = Quantizer(
        bits=bits,
        scheme=metadata["scheme"],
        granularity=metadata["granularity"],
        group_size=metadata.get("group_size") or 128,
        axis=metadata.get("axis") if metadata.get("axis") is not None else 0,
    )

    params = {
        "scale": scales,
        "granularity": metadata["granularity"],
        "bits": bits,
    }

    if metadata["granularity"] in ("per_channel", "per_group"):
        params["axis"] = metadata["axis"]

    if metadata["granularity"] == "per_group":
        params["group_size"] = metadata["group_size"]

    reconstructed = quantizer.dequantize(q_tensor, params)

    expected_shape = tuple(metadata["original_shape"])
    if tuple(reconstructed.shape) != expected_shape:
        raise ValueError(
            f"Reconstructed shape {tuple(reconstructed.shape)} "
            f"does not match metadata shape {expected_shape}"
        )

    return reconstructed.to(_DTYPE_MAP[metadata["original_dtype"]])
