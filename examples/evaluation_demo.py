
import torch
import torch.nn as nn

from quant_engine.evaluation import evaluate_quantization


class EvaluationCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def print_results(results):

    print("\n" + "=" * 55)
    print(f"INT{results['bits']} EVALUATION")
    print("=" * 55)

    print(
        f"Original storage: "
        f"{results['original_storage_bytes'] / 1024:.2f} KB"
    )

    print(
        f"Quantized storage: "
        f"{results['quantized_storage_bytes'] / 1024:.2f} KB"
    )

    print(
        f"Storage reduction: "
        f"{results['storage_reduction_percent']:.2f}%"
    )

    print("\nOutput metrics")

    for metric, value in results["output_metrics"].items():
        print(f"{metric}: {value:.8f}")

    print("\nLatency")

    print(
        f"Quantization: "
        f"{results['quantization_latency_ms']:.4f} ms"
    )

    print(
        f"Dequantization: "
        f"{results['dequantization_latency_ms']:.4f} ms"
    )

    print(
        f"FP32 inference: "
        f"{results['original_inference_latency_ms']:.4f} ms"
    )

    print(
        f"Quantized inference: "
        f"{results['quantized_inference_latency_ms']:.4f} ms"
    )

    print("\nLayer-wise weight errors")

    for name, metrics in results["layerwise_metrics"].items():

        print(f"\n{name}")

        for metric, value in metrics.items():
            print(f"  {metric}: {value:.8f}")


if __name__ == "__main__":

    torch.manual_seed(42)

    model = EvaluationCNN()

    inputs = torch.randn(8, 3, 64, 64)

    for bits in [8, 4]:

        results = evaluate_quantization(
            model,
            inputs,
            bits=bits,
            granularity="per_channel",
            axis=0,
            warmup=5,
            iterations=30
        )

        print_results(results)
