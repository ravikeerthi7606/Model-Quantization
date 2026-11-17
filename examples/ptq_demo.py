
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from quant_engine.ptq import ptq_quantize_model


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
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


def main():

    torch.manual_seed(42)

    model = CNN()
    model.eval()

    images = torch.randn(100, 3, 32, 32)
    labels = torch.zeros(100, dtype=torch.long)

    dataset = TensorDataset(images, labels)

    loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False
    )

    test_input = torch.randn(8, 3, 32, 32)

    with torch.no_grad():
        original_output = model(test_input)

    for bits in [8, 4]:

        ptq_model, calibration_results = ptq_quantize_model(
            model,
            loader,
            bits=bits,
            granularity="per_channel",
            axis=0
        )

        ptq_model.eval()

        with torch.no_grad():
            quantized_output = ptq_model(test_input)

        mse = torch.mean(
            (original_output - quantized_output) ** 2
        ).item()

        print(f"\nINT{bits} PTQ RESULTS")
        print("-" * 40)

        print(f"Output shape: {quantized_output.shape}")
        print(f"Output MSE: {mse:.8f}")

        print(
            f"Calibrated layers: {len(calibration_results)}"
        )


if __name__ == "__main__":
    main()
