
import torch
import torch.nn as nn

from quant_engine.model_quantization import quantize_model

torch.manual_seed(42)

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(32 * 32 * 32, 10)
)

x = torch.randn(4, 3, 32, 32)

original_output = model(x)

print("Original output:", original_output.shape)

for bits in [8, 4]:

    quantized_model = quantize_model(
        model,
        bits=bits,
        granularity="per_channel",
        axis=0
    )

    quantized_output = quantized_model(x)

    mse = torch.mean(
        (original_output - quantized_output) ** 2
    )

    print(f"\nINT{bits}")
    print("Output shape:", quantized_output.shape)
    print("Output MSE:", mse.item())
