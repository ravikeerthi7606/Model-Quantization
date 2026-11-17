
import torch
import torch.nn as nn

from quant_engine.model_quantization import quantize_model


def create_model():
    return nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )


def test_original_model_unchanged():
    model = create_model()

    original_weights = [
        p.detach().clone()
        for p in model.parameters()
    ]

    quantize_model(
        model,
        bits=4,
        granularity="per_channel"
    )

    for original, current in zip(
        original_weights, model.parameters()
    ):
        assert torch.equal(original, current)


def test_bias_unchanged():
    model = create_model()

    original_biases = [
        layer.bias.detach().clone()
        for layer in model.modules()
        if isinstance(layer, nn.Linear)
    ]

    quantized = quantize_model(
        model,
        bits=4,
        granularity="per_channel"
    )

    new_biases = [
        layer.bias
        for layer in quantized.modules()
        if isinstance(layer, nn.Linear)
    ]

    for original, new in zip(original_biases, new_biases):
        assert torch.equal(original, new)


def test_output_shape():
    model = create_model()
    x = torch.randn(32, 128)

    quantized = quantize_model(
        model,
        bits=4,
        granularity="per_channel"
    )

    assert model(x).shape == quantized(x).shape


def test_int8_and_int4():
    model = create_model()

    for bits in [4, 8]:
        quantized = quantize_model(
            model,
            bits=bits,
            granularity="per_channel"
        )

        assert isinstance(quantized, nn.Module)


def test_model_without_linear():
    model = nn.ReLU()

    quantized = quantize_model(
        model,
        bits=8,
        granularity="per_channel"
    )

    assert isinstance(quantized, nn.Module)
