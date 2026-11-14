
from pathlib import Path
import math
import torch


FORMAT_VERSION = 1


def save_quantized(payload, scales, metadata, path):
    """
    Save quantized payload, scales, and reconstruction metadata.
    """

    if not isinstance(payload, torch.Tensor):
        raise TypeError("payload must be a torch.Tensor")

    if not isinstance(scales, torch.Tensor):
        scales = torch.as_tensor(scales)

    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")

    required_fields = [
        "bits",
        "scheme",
        "granularity",
        "original_shape",
        "original_numel",
        "original_dtype",
    ]

    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Missing metadata field: {field}")

    shape = tuple(metadata["original_shape"])
    numel = math.prod(shape)

    if numel != metadata["original_numel"]:
        raise ValueError("original_numel does not match original_shape")

    artifact = {
        "format_version": FORMAT_VERSION,
        "payload": payload.detach().cpu(),
        "scales": scales.detach().cpu(),
        "metadata": metadata,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(artifact, path)


def load_quantized(path, map_location="cpu"):
    """
    Load a saved quantized artifact.
    """

    artifact = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )

    if artifact["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported serialization format version")

    return (
        artifact["payload"],
        artifact["scales"],
        artifact["metadata"],
    )
