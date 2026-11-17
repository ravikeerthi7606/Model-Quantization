import torch
import torch.nn as nn

from quant_engine.model_quantization import quantize_model


torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

x = torch.randn(32, 128)

original_output = model(x)

int8_model = quantize_model(
    model,
    bits=8,
    granularity="per_channel"
)

int4_model = quantize_model(
    model,
    bits=4,
    granularity="per_channel"
)

int8_output = int8_model(x)
int4_output = int4_model(x)

print("FP32 vs INT8 MSE:",
      torch.mean((original_output - int8_output) ** 2).item())

print("FP32 vs INT4 MSE:",
      torch.mean((original_output - int4_output) ** 2).item())