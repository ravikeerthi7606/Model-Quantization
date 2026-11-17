
import pytest
import torch
import torch.nn as nn

from quant_engine.observers import (
    MinMaxObserver,
    ModelCalibrator
)

from quant_engine.observers import PercentileObserver


def test_minmax_observer():

    observer = MinMaxObserver(bits=8)

    observer.update(torch.tensor([-2.0, 1.0]))
    observer.update(torch.tensor([-0.5, 4.0]))

    stats = observer.get_statistics()

    assert stats["min"] == -2.0
    assert stats["max"] == 4.0
    assert stats["num_batches"] == 2
    assert stats["num_values"] == 4


@pytest.mark.parametrize("bits,expected_qmax", [
    (4, 7),
    (8, 127)
])
def test_calculate_qparams(bits, expected_qmax):

    observer = MinMaxObserver(bits=bits)

    observer.update(torch.tensor([-2.0, 4.0]))

    scale, zero_point = observer.calculate_qparams()

    assert torch.isclose(
        scale,
        torch.tensor(4.0 / expected_qmax)
    )

    assert zero_point == 0


def test_zero_tensor():

    observer = MinMaxObserver(bits=8)

    observer.update(torch.zeros(10))

    scale, zero_point = observer.calculate_qparams()

    assert scale.item() == 1.0
    assert zero_point == 0


def test_empty_tensor_rejected():

    observer = MinMaxObserver()

    with pytest.raises(ValueError):
        observer.update(torch.tensor([]))


def test_nan_rejected():

    observer = MinMaxObserver()

    with pytest.raises(ValueError):
        observer.update(torch.tensor([1.0, float("nan")]))


def test_reset():

    observer = MinMaxObserver()

    observer.update(torch.tensor([1.0, 2.0]))
    observer.reset()

    assert observer.min_val is None
    assert observer.max_val is None
    assert observer.num_batches == 0
    assert observer.num_values == 0


def test_calibrator():

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

    calibration_data = [
        (torch.randn(4, 4), torch.zeros(4)),
        (torch.randn(4, 4), torch.zeros(4)),
        (torch.randn(4, 4), torch.zeros(4))
    ]

    calibrator = ModelCalibrator(bits=8)

    observers = calibrator.calibrate(
        model,
        calibration_data
    )

    assert "0" in observers
    assert "1" in observers
    assert "2" in observers

    assert observers["0"].num_batches == 3
    assert observers["1"].num_batches == 3
    assert observers["2"].num_batches == 3


def test_training_state_restored():

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.Dropout(0.5),
        nn.Linear(8, 2)
    )

    model.train()
    model[1].eval()

    original_states = [
        module.training
        for module in model.modules()
    ]

    calibrator = ModelCalibrator()

    calibrator.calibrate(
        model,
        [torch.randn(4, 4)]
    )

    restored_states = [
        module.training
        for module in model.modules()
    ]

    assert original_states == restored_states


def test_hooks_removed():

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU()
    )

    calibrator = ModelCalibrator()

    calibrator.calibrate(
        model,
        [torch.randn(4, 4)]
    )

    for module in model.modules():
        assert len(module._forward_hooks) == 0


def test_percentile_observer():

    observer = PercentileObserver(
        bits=8,
        percentile=99.0,
        max_samples=10000
    )

    values = torch.cat([
        torch.ones(999),
        torch.tensor([1000.0])
    ])

    observer.update(values)

    scale, zero_point = observer.calculate_qparams()

    assert torch.isclose(
        scale,
        torch.tensor(1.0 / 127)
    )

    assert zero_point == 0


def test_percentile_clipping():

    observer = PercentileObserver(
        bits=8,
        percentile=90.0
    )

    observer.update(torch.tensor([
        1.0, 2.0, 3.0, 4.0, 100.0
    ]))

    scale, _ = observer.calculate_qparams()

    assert observer.clipping_threshold < 100.0
    assert scale.item() > 0


def test_percentile_sample_limit():

    observer = PercentileObserver(
        bits=8,
        percentile=99.9,
        max_samples=1000,
        samples_per_update=5000
    )

    observer.update(torch.randn(10000))

    assert observer.samples.numel() <= 1000


def test_invalid_percentile():

    with pytest.raises(ValueError):

        PercentileObserver(
            bits=8,
            percentile=101
        )


def test_percentile_reset():

    observer = PercentileObserver(bits=8)

    observer.update(torch.randn(100))
    observer.reset()

    assert observer.samples.numel() == 0
    assert observer.min_val is None
    assert observer.max_val is None
