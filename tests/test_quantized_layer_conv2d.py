
import pytest
import torch
import torch.nn as nn

from quant_engine.quantized_layer import QuantizedConv2d


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize(
    "granularity",
    ["per_tensor", "per_channel", "per_group"]
)
def test_quantized_conv2d_forward(bits, granularity):

    torch.manual_seed(42)

    layer = nn.Conv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        padding=1
    )

    quantized_layer = QuantizedConv2d.from_float(
        layer,
        bits=bits,
        granularity=granularity,
        group_size=16,
        axis=0
    )

    x = torch.randn(4, 3, 32, 32)

    output = quantized_layer(x)

    assert output.shape == (4, 16, 32, 32)
    assert torch.isfinite(output).all()


def test_int4_packed_storage():

    layer = nn.Conv2d(3, 16, 3)

    quantized_layer = QuantizedConv2d.from_float(
        layer,
        bits=4
    )

    assert quantized_layer.weight.dtype == torch.uint8

    assert quantized_layer.weight.numel() == (
        layer.weight.numel() + 1
    ) // 2


def test_int8_storage():

    layer = nn.Conv2d(3, 16, 3)

    quantized_layer = QuantizedConv2d.from_float(
        layer,
        bits=8
    )

    assert quantized_layer.weight.dtype == torch.int8


def test_original_layer_unchanged():

    layer = nn.Conv2d(3, 16, 3)

    original_weight = layer.weight.detach().clone()
    original_bias = layer.bias.detach().clone()

    QuantizedConv2d.from_float(layer, bits=4)

    assert torch.equal(layer.weight, original_weight)
    assert torch.equal(layer.bias, original_bias)


def test_grouped_convolution():

    layer = nn.Conv2d(
        8,
        16,
        kernel_size=3,
        padding=1,
        groups=2
    )

    quantized_layer = QuantizedConv2d.from_float(
        layer,
        bits=8
    )

    x = torch.randn(2, 8, 32, 32)

    output = quantized_layer(x)

    assert output.shape == (2, 16, 32, 32)


def test_bias_none():

    layer = nn.Conv2d(
        3,
        16,
        kernel_size=3,
        bias=False
    )

    quantized_layer = QuantizedConv2d.from_float(
        layer,
        bits=8
    )

    x = torch.randn(2, 3, 32, 32)

    output = quantized_layer(x)

    assert output.shape == (2, 16, 30, 30)
    assert quantized_layer.bias is None
