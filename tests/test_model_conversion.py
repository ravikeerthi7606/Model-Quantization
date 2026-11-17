
import torch
import torch.nn as nn

from quant_engine.model_conversion import quantize_model
from quant_engine.quantized_layer import (
    QuantizedLinear,
    QuantizedConv2d
)


def test_model_layer_replacement():

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 30 * 30, 10)
    )

    quantized_model = quantize_model(
        model,
        bits=4
    )

    assert isinstance(
        quantized_model[0],
        QuantizedConv2d
    )

    assert isinstance(
        quantized_model[3],
        QuantizedLinear
    )


def test_model_forward():

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 30 * 30, 10)
    )

    quantized_model = quantize_model(
        model,
        bits=8
    )

    x = torch.randn(2, 3, 32, 32)

    output = quantized_model(x)

    assert output.shape == (2, 10)
    assert torch.isfinite(output).all()


def test_original_model_unchanged():

    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU()
    )

    original_weight = model[0].weight.detach().clone()

    quantize_model(model, bits=4)

    assert torch.equal(
        model[0].weight,
        original_weight
    )


def test_unsupported_layers_preserved():

    model = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 16)
    )

    quantized_model = quantize_model(model)

    assert isinstance(quantized_model[0], nn.ReLU)
    assert isinstance(quantized_model[1], nn.Dropout)
    assert isinstance(quantized_model[2], QuantizedLinear)


def test_nested_model():

    model = nn.Sequential(
        nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU()
        ),
        nn.Linear(16, 4)
    )

    quantized_model = quantize_model(model)

    assert isinstance(
        quantized_model[0][0],
        QuantizedLinear
    )

    assert isinstance(
        quantized_model[1],
        QuantizedLinear
    )
