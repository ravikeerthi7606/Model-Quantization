
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

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
        x = self.features(x)
        return self.classifier(x)



if __name__ == "__main__":

    torch.manual_seed(42)

    model = CNN()

    # Create sample calibration dataset
    images = torch.randn(100, 3, 32, 32)
    labels = torch.randint(0, 10, (100,))

    dataset = TensorDataset(images, labels)

    loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False
    )

    calibrator = ModelCalibrator(bits=8)

    calibrator.calibrate(
        model,
        loader
    )

    results = calibrator.get_calibration_results()

    print("\nCALIBRATION RESULTS")
    print("=" * 55)

    for name, result in results.items():

        print(f"\nLayer: {name}")
        print(f"Min: {result['min']:.6f}")
        print(f"Max: {result['max']:.6f}")
        print(f"Scale: {result['scale'].item():.8f}")
        print(f"Zero point: {result['zero_point']}")
        print(f"Observed batches: {result['num_batches']}")
        print(f"Observed values: {result['num_values']}")
