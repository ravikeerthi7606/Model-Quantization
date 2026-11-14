
import torch
import pytest

from quant_engine.metrics import compute_metrics


def test_metrics():

    original = torch.tensor([1.0, 2.0, 3.0])

    reconstructed = torch.tensor([1.1, 1.9, 3.0])

    metrics = compute_metrics(original, reconstructed)

    assert metrics["mse"] == pytest.approx(0.0066667, rel=1e-4)
    assert metrics["mae"] == pytest.approx(0.0666667, rel=1e-4)
    assert metrics["max_abs_error"] == pytest.approx(0.1)


def test_perfect_reconstruction():

    x = torch.randn(100)

    metrics = compute_metrics(x, x)

    assert metrics["mse"] == 0
    assert metrics["mae"] == 0
    assert metrics["max_abs_error"] == 0
    assert metrics["sqnr_db"] == float("inf")


def test_shape_mismatch():

    x = torch.randn(10)
    y = torch.randn(5)

    with pytest.raises(ValueError):
        compute_metrics(x, y)


def test_nan_rejection():

    x = torch.tensor([1.0, float("nan")])
    y = torch.tensor([1.0, 2.0])

    with pytest.raises(ValueError):
        compute_metrics(x, y)


def test_zero_signal():

    x = torch.zeros(10)
    y = torch.ones(10)

    metrics = compute_metrics(x, y)

    assert metrics["sqnr_db"] == float("-inf")
