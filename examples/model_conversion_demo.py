
import torch
import torch.nn as nn

from quant_engine.model_conversion import quantize_model


# ---------------------------------
# Example 1: MLP architecture
# ---------------------------------

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------------
# Example 2: CNN architecture
# ---------------------------------

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------
# Storage calculation
# ---------------------------------

def get_model_size(model):

    total_bytes = 0

    for tensor in list(model.parameters()) + list(model.buffers()):
        total_bytes += tensor.numel() * tensor.element_size()

    return total_bytes


# ---------------------------------
# Evaluation
# ---------------------------------

def evaluate_model(model, quantized_model, x, bits):

    model.eval()
    quantized_model.eval()

    with torch.no_grad():

        original_output = model(x)
        quantized_output = quantized_model(x)

    mse = torch.mean(
        (original_output - quantized_output) ** 2
    ).item()

    max_error = torch.max(
        torch.abs(original_output - quantized_output)
    ).item()

    original_size = get_model_size(model)
    quantized_size = get_model_size(quantized_model)

    reduction = (
        1 - quantized_size / original_size
    ) * 100

    print(f"\nINT{bits} RESULTS")
    print("-" * 45)

    print(f"Original storage : {original_size / 1024:.2f} KB")
    print(f"Quantized storage: {quantized_size / 1024:.2f} KB")
    print(f"Storage reduction: {reduction:.2f}%")

    print(f"Output MSE       : {mse:.8f}")
    print(f"Maximum error    : {max_error:.8f}")

    return {
        "bits": bits,
        "mse": mse,
        "max_error": max_error,
        "original_size": original_size,
        "quantized_size": quantized_size,
        "reduction": reduction
    }


# ---------------------------------
# Run experiments
# ---------------------------------

def run_experiment(name, model, x):

    print("\n" + "=" * 55)
    print(f"MODEL: {name}")
    print("=" * 55)

    for bits in [8, 4]:

        quantized_model = quantize_model(
            model,
            bits=bits,
            granularity="per_channel",
            axis=0
        )

        evaluate_model(
            model,
            quantized_model,
            x,
            bits
        )


if __name__ == "__main__":

    torch.manual_seed(42)

    mlp = MLP()
    mlp_input = torch.randn(32, 128)

    cnn = CNN()
    cnn_input = torch.randn(8, 3, 32, 32)

    run_experiment(
        "MLP",
        mlp,
        mlp_input
    )

    run_experiment(
        "CNN",
        cnn,
        cnn_input
    )
