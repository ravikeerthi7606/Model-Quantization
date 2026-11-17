
import time
import torch
import torch.nn as nn

from .quantized_layer import QuantizedLinear, QuantizedConv2d
from .model_conversion import quantize_model


def calculate_metrics(original, reconstructed):

    if original.shape != reconstructed.shape:
        raise ValueError("Tensor shapes must match")

    original = original.float()
    reconstructed = reconstructed.float()

    error = original - reconstructed

    mse = torch.mean(error ** 2).item()
    mae = torch.mean(torch.abs(error)).item()
    max_error = torch.max(torch.abs(error)).item()

    signal_power = torch.mean(original ** 2).item()
    noise_power = mse

    if noise_power == 0:
        sqnr = float("inf")
    elif signal_power == 0:
        sqnr = float("-inf")
    else:
        sqnr = 10 * torch.log10(
            torch.tensor(signal_power / noise_power)
        ).item()

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "sqnr_db": sqnr
    }


def get_model_storage(model):

    total_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
    )

    return total_bytes


def get_layerwise_metrics(original_model, quantized_model):

    original_layers = dict(original_model.named_modules())
    quantized_layers = dict(quantized_model.named_modules())

    results = {}

    for name, original_layer in original_layers.items():

        quantized_layer = quantized_layers.get(name)

        if isinstance(
            quantized_layer,
            (QuantizedLinear, QuantizedConv2d)
        ):

            reconstructed_weight = (
                quantized_layer.dequantize_weight()
            )

            results[name] = calculate_metrics(
                original_layer.weight.detach(),
                reconstructed_weight
            )

    return results


def synchronize(device):

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_callable(
    function,
    device,
    warmup=2,
    iterations=10
):

    for _ in range(warmup):
        function()

    synchronize(device)

    start = time.perf_counter()

    for _ in range(iterations):
        function()

    synchronize(device)

    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000


def benchmark_inference(
    model,
    inputs,
    warmup=10,
    iterations=100
):

    model.eval()

    device = inputs.device

    with torch.no_grad():

        for _ in range(warmup):
            model(inputs)

        synchronize(device)

        start = time.perf_counter()

        for _ in range(iterations):
            model(inputs)

        synchronize(device)

    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000


def benchmark_dequantization(
    model,
    device,
    warmup=2,
    iterations=10
):

    layers = [
        module
        for module in model.modules()
        if isinstance(
            module,
            (QuantizedLinear, QuantizedConv2d)
        )
    ]

    if not layers:
        return 0.0

    def dequantize_all():
        for layer in layers:
            layer.dequantize_weight()

    return benchmark_callable(
        dequantize_all,
        device,
        warmup,
        iterations
    )


def evaluate_quantization(
    model,
    inputs,
    bits=8,
    granularity="per_channel",
    group_size=128,
    axis=0,
    warmup=10,
    iterations=100
):

    model.eval()

    device = inputs.device

    # Measure model conversion / quantization latency
    quantization_latency = benchmark_callable(
        lambda: quantize_model(
            model,
            bits=bits,
            granularity=granularity,
            group_size=group_size,
            axis=axis
        ),
        device,
        warmup=1,
        iterations=5
    )

    # Create the model used for evaluation
    quantized_model = quantize_model(
        model,
        bits=bits,
        granularity=granularity,
        group_size=group_size,
        axis=axis
    )

    quantized_model.eval()

    with torch.no_grad():

        original_output = model(inputs)
        quantized_output = quantized_model(inputs)

    output_metrics = calculate_metrics(
        original_output,
        quantized_output
    )

    original_storage = get_model_storage(model)
    quantized_storage = get_model_storage(quantized_model)

    reduction = (
        1 - quantized_storage / original_storage
    ) * 100 if original_storage else 0.0

    original_latency = benchmark_inference(
        model,
        inputs,
        warmup,
        iterations
    )

    quantized_latency = benchmark_inference(
        quantized_model,
        inputs,
        warmup,
        iterations
    )

    dequantization_latency = benchmark_dequantization(
        quantized_model,
        device
    )

    layerwise_metrics = get_layerwise_metrics(
        model,
        quantized_model
    )

    return {
        "bits": bits,
        "granularity": granularity,
        "original_storage_bytes": original_storage,
        "quantized_storage_bytes": quantized_storage,
        "storage_reduction_percent": reduction,
        "output_metrics": output_metrics,
        "layerwise_metrics": layerwise_metrics,
        "quantization_latency_ms": quantization_latency,
        "dequantization_latency_ms": dequantization_latency,
        "original_inference_latency_ms": original_latency,
        "quantized_inference_latency_ms": quantized_latency,
        "quantized_model": quantized_model
    }
