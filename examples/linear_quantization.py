
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_engine import Quantizer

torch.manual_seed(42)

layer = nn.Linear(128, 256)
x = torch.randn(32, 128)

# FP32 reference
original_output = layer(x)

results = {}

for bits in [8, 4]:
    quantizer = Quantizer(
        bits=bits,
        granularity="per_channel",
        axis=0
    )

    q_weights, params = quantizer.quantize(layer.weight)

    reconstructed_weights = quantizer.dequantize(
        q_weights, params
    )

    output = F.linear(
        x,
        reconstructed_weights,
        layer.bias
    )

    mse = torch.mean(
        (original_output - output) ** 2
    ).item()

    max_error = torch.max(
        torch.abs(original_output - output)
    ).item()

    results[bits] = {
        "mse": mse,
        "max_error": max_error
    }

print("\nMODEL OUTPUT COMPARISON")
print("=" * 45)
print("FP32: Reference output")

for bits, metrics in results.items():
    print(f"\nINT{bits}")
    print(f"MSE: {metrics['mse']:.8f}")
    print(f"Maximum error: {metrics['max_error']:.8f}")
