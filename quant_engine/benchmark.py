
import time
import statistics
import torch

from .quantizer import Quantizer
from .packing import pack_int4, unpack_int4
from .metrics import compute_metrics


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(fn, device, warmup, iterations):

    for _ in range(warmup):
        result = fn()

    _synchronize(device)

    timings = []

    for _ in range(iterations):
        start = time.perf_counter()

        result = fn()

        _synchronize(device)

        timings.append(
            (time.perf_counter() - start) * 1000
        )

    return result, statistics.median(timings)


def benchmark_quantization(
    tensor,
    bits=8,
    granularity="per_tensor",
    group_size=128,
    axis=0,
    warmup=5,
    iterations=20
):

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    if tensor.numel() == 0:
        raise ValueError("Input tensor cannot be empty")

    if warmup < 0 or iterations <= 0:
        raise ValueError("Invalid benchmark iteration count")

    quantizer = Quantizer(
        bits=bits,
        granularity=granularity,
        group_size=group_size,
        axis=axis
    )

    device = tensor.device

    # Encode: quantization + optional INT4 packing
    def encode():

        q_tensor, params = quantizer.quantize(tensor)

        if bits == 4:
            payload, packing_metadata = pack_int4(q_tensor)
        else:
            payload = q_tensor
            packing_metadata = None

        return payload, params, packing_metadata

    encoded, quantize_ms = _measure(
        encode, device, warmup, iterations
    )

    payload, params, packing_metadata = encoded

    # Decode: optional unpacking + dequantization
    def decode():

        if bits == 4:
            q_tensor = unpack_int4(
                payload,
                packing_metadata
            )
        else:
            q_tensor = payload

        return quantizer.dequantize(q_tensor, params)

    reconstructed, dequantize_ms = _measure(
        decode, device, warmup, iterations
    )

    # Storage calculations
    original_bytes = (
        tensor.numel() * tensor.element_size()
    )

    payload_bytes = (
        payload.numel() * payload.element_size()
    )

    scale = params["scale"]

    scale_bytes = (
        scale.numel() * scale.element_size()
    )

    compressed_bytes = payload_bytes + scale_bytes

    compression_ratio = original_bytes / compressed_bytes

    memory_reduction = (
        1 - compressed_bytes / original_bytes
    ) * 100

    # Accuracy metrics
    metrics = compute_metrics(
        tensor,
        reconstructed
    )

    return {
        "bits": bits,
        "granularity": granularity,
        "device": str(device),

        "quantize_ms": quantize_ms,
        "dequantize_ms": dequantize_ms,
        "total_ms": quantize_ms + dequantize_ms,

        "original_bytes": original_bytes,
        "payload_bytes": payload_bytes,
        "scale_bytes": scale_bytes,
        "compressed_bytes": compressed_bytes,

        "compression_ratio": compression_ratio,
        "memory_reduction_pct": memory_reduction,

        "metrics": metrics
    }
