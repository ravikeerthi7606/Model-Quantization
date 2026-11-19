
import torch
import torch.nn as nn

from quant_engine.observers import ModelCalibrator




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
        return self.classifier(
            self.features(x)
        )


if __name__ == "__main__":

    torch.manual_seed(42)

    model = CNN()

    calibration_data = [
        torch.randn(10, 3, 32, 32)
        for _ in range(10)
    ]

    minmax = ModelCalibrator(
        bits=8,
        observer_type="minmax"
    )

    percentile = ModelCalibrator(
        bits=8,
        observer_type="percentile",
        percentile=99.9
    )

    minmax.calibrate(model, calibration_data)

    percentile.calibrate(model, calibration_data)

    minmax_results = minmax.get_calibration_results()
    percentile_results = percentile.get_calibration_results()

    print("\nCALIBRATION COMPARISON")
    print("=" * 75)

    for name in minmax_results:

        print(f"\nLayer: {name}")

        print(
            f"MinMax scale: "
            f"{minmax_results[name]['scale'].item():.8f}"
        )

        print(
            f"Percentile scale: "
            f"{percentile_results[name]['scale'].item():.8f}"
        )

        print(
            f"Percentile clipping threshold: "
            f"{percentile_results[name]['clipping_threshold']:.8f}"
        )
