
import pytest
import torch
import torch.nn as nn

from quant_engine.quantized_layer import QuantizedLinear


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize(
    "granularity",
    ["per_tensor", "per_channel", "per_group"]
)
def test_quantized_linear_forward(bits, granularity):

    torch.manual_seed(42)

    layer = nn.Linear(128, 64)

    quantized_layer = QuantizedLinear.from_float(
        layer,
        bits=bits,
        granularity=granularity,
        group_size=32,
        axis=0
    )

    x = torch.randn(16, 128)

    output = quantized_layer(x)

    assert output.shape == (16, 64)
    assert torch.isfinite(output).all()


def test_int4_uses_packed_storage():

    layer = nn.Linear(128, 64)

    quantized_layer = QuantizedLinear.from_float(
        layer,
        bits=4
    )

    assert quantized_layer.weight.dtype == torch.uint8

    assert quantized_layer.weight.numel() == (
        layer.weight.numel() + 1
    ) // 2


def test_int8_uses_int8_storage():

    layer = nn.Linear(128, 64)

    quantized_layer = QuantizedLinear.from_float(
        layer,
        bits=8
    )

    assert quantized_layer.weight.dtype == torch.int8


def test_original_layer_is_not_modified():

    layer = nn.Linear(128, 64)

    original_weight = layer.weight.detach().clone()

    QuantizedLinear.from_float(
        layer,
        bits=4
    )

    assert torch.equal(
        layer.weight,
        original_weight
    )


def test_bias_none():

    layer = nn.Linear(128, 64, bias=False)

    quantized_layer = QuantizedLinear.from_float(
        layer,
        bits=8
    )

    x = torch.randn(4, 128)

    output = quantized_layer(x)

    assert output.shape == (4, 64)
    assert quantized_layer.bias is None
