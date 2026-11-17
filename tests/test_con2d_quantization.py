
import torch
import torch.nn as nn

from quant_engine.model_quantization import quantize_model


def create_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1)
    )


def test_conv2d_output_shape():
    model = create_model()
    x = torch.randn(2, 3, 32, 32)

    quantized = quantize_model(
        model, bits=8, granularity="per_channel"
    )

    assert model(x).shape == quantized(x).shape


def test_conv2d_original_unchanged():
    model = create_model()

    original_weights = [
        layer.weight.detach().clone()
        for layer in model.modules()
        if isinstance(layer, nn.Conv2d)
    ]

    quantize_model(model, bits=4)

    new_weights = [
        layer.weight
        for layer in model.modules()
        if isinstance(layer, nn.Conv2d)
    ]

    for original, current in zip(original_weights, new_weights):
        assert torch.equal(original, current)


def test_conv2d_bias_unchanged():
    model = create_model()

    quantized = quantize_model(model, bits=4)

    original_biases = [
        layer.bias for layer in model.modules()
        if isinstance(layer, nn.Conv2d)
    ]

    quantized_biases = [
        layer.bias for layer in quantized.modules()
        if isinstance(layer, nn.Conv2d)
    ]

    for original, current in zip(original_biases, quantized_biases):
        assert torch.equal(original, current)


def test_conv2d_int4_int8():
    model = create_model()

    for bits in [4, 8]:
        quantized = quantize_model(
            model,
            bits=bits,
            granularity="per_channel"
        )

        assert isinstance(quantized, nn.Module)
