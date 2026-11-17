
import pytest
import torch
import torch.nn as nn

from quant_engine.ptq import (
    ActivationFakeQuantizer,
    ptq_quantize_model
)

from quant_engine.quantized_layer import QuantizedLinear


@pytest.mark.parametrize("bits", [4, 8])
def test_ptq_forward(bits):

    torch.manual_seed(42)

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

    calibration_data = [
        torch.randn(8, 4)
        for _ in range(5)
    ]

    quantized_model, results = ptq_quantize_model(
        model,
        calibration_data,
        bits=bits
    )

    x = torch.randn(4, 4)

    output = quantized_model(x)

    assert output.shape == (4, 2)
    assert torch.isfinite(output).all()

    assert "0" in results
    assert "1" in results
    assert "2" in results


def test_weight_quantization_and_activation_quantization():

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

    calibration_data = [
        torch.randn(8, 4)
        for _ in range(3)
    ]

    quantized_model, _ = ptq_quantize_model(
        model,
        calibration_data,
        bits=4
    )

    assert isinstance(quantized_model[0], nn.Sequential)

    assert isinstance(
        quantized_model[0][0],
        QuantizedLinear
    )

    assert isinstance(
        quantized_model[0][1],
        ActivationFakeQuantizer
    )


def test_original_model_unchanged():

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

    original_weight = model[0].weight.detach().clone()

    calibration_data = [
        torch.randn(8, 4)
        for _ in range(3)
    ]

    ptq_quantize_model(
        model,
        calibration_data,
        bits=8
    )

    assert torch.equal(
        model[0].weight,
        original_weight
    )


def test_activation_quantizer():

    quantizer = ActivationFakeQuantizer(
        scale=0.1,
        bits=8
    )

    x = torch.tensor([0.14, 0.26, -0.34])

    output = quantizer(x)

    expected = torch.tensor([0.1, 0.3, -0.3])

    assert torch.allclose(output, expected)


def test_activation_clamping():

    quantizer = ActivationFakeQuantizer(
        scale=0.1,
        bits=8
    )

    x = torch.tensor([100.0, -100.0])

    output = quantizer(x)

    assert output.max() <= 12.7
    assert output.min() >= -12.8
