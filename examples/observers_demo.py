
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


class PercentileObserver(MinMaxObserver):

    def __init__(
        self,
        bits=8,
        percentile=99.9,
        max_samples=200000,
        samples_per_update=20000
    ):

        super().__init__(bits=bits)

        if not isinstance(percentile, (int, float)):
            raise TypeError("percentile must be numeric")

        if not 0 < percentile <= 100:
            raise ValueError("percentile must be in (0, 100]")

        if not isinstance(max_samples, int) or max_samples <= 0:
            raise ValueError("max_samples must be positive")

        if not isinstance(samples_per_update, int) or samples_per_update <= 0:
            raise ValueError("samples_per_update must be positive")

        self.percentile = percentile
        self.max_samples = max_samples
        self.samples_per_update = samples_per_update

        self.samples = torch.empty(0, dtype=torch.float32)
        self.clipping_threshold = None

    def update(self, tensor):

        # Maintain global min/max and observation counts
        super().update(tensor)

        values = tensor.detach().abs().reshape(-1)

        # Sample large activation tensors to limit memory usage
        if values.numel() > self.samples_per_update:

            indices = torch.randperm(
                values.numel(),
                device=values.device
            )[:self.samples_per_update]

            values = values[indices]

        values = values.float().cpu()

        combined = torch.cat([
            self.samples,
            values
        ])

        # Keep a bounded random sample
        if combined.numel() > self.max_samples:

            indices = torch.randperm(
                combined.numel()
            )[:self.max_samples]

            combined = combined[indices]

        self.samples = combined

    def calculate_qparams(self):

        if self.samples.numel() == 0:
            raise RuntimeError(
                "Observer has not collected any data"
            )

        threshold = torch.quantile(
            self.samples,
            self.percentile / 100.0
        )

        self.clipping_threshold = threshold.item()

        scale = (
            threshold / self.q_max
            if threshold.item() != 0
            else torch.tensor(1.0)
        )

        return scale.float(), 0

    def get_statistics(self):

        stats = super().get_statistics()

        stats.update({
            "percentile": self.percentile,
            "clipping_threshold": self.clipping_threshold,
            "sample_count": self.samples.numel()
        })

        return stats

    def reset(self):

        super().reset()

        self.samples = torch.empty(
            0,
            dtype=torch.float32
        )

        self.clipping_threshold = None


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
