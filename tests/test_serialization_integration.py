import pytest
import torch

from quant_engine.quantizer import Quantizer
from quant_engine.serialization import (
    quantize_and_save,
    load_and_dequantize,
    load_quantized,
)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize(
    "granularity,kwargs",
    [
        ("per_tensor", {}),
        ("per_channel", {"axis": 0}),
        ("per_group", {"axis": -1, "group_size": 4}),
    ],
)
def test_quantize_save_load_dequantize_round_trip(
    tmp_path, bits, granularity, kwargs
):
    torch.manual_seed(7)
    x = torch.randn(3, 7, dtype=torch.float32)

    quantizer = Quantizer(
        bits=bits,
        granularity=granularity,
        **kwargs,
    )

    path = tmp_path / f"{bits}_{granularity}.pt"
    quantize_and_save(x, quantizer, path)
    reconstructed = load_and_dequantize(path)

    assert reconstructed.shape == x.shape
    assert reconstructed.dtype == x.dtype
    assert torch.isfinite(reconstructed).all()

    # The saved/reloaded reconstruction should match direct quantizer output.
    q, params = quantizer.quantize(x)
    direct = quantizer.dequantize(q, params)
    torch.testing.assert_close(reconstructed, direct)


def test_int4_odd_numel_round_trip(tmp_path):
    torch.manual_seed(11)
    x = torch.randn(5, dtype=torch.float32)
    quantizer = Quantizer(bits=4, granularity="per_tensor")

    path = tmp_path / "odd_int4.pt"
    quantize_and_save(x, quantizer, path)

    payload, scales, metadata = load_quantized(path)

    assert payload.dtype == torch.uint8
    assert payload.numel() == 3
    assert metadata["original_numel"] == 5

    reconstructed = load_and_dequantize(path)
    assert reconstructed.shape == (5,)
    assert reconstructed.dtype == x.dtype


def test_int8_payload_is_not_packed(tmp_path):
    x = torch.randn(2, 3)
    quantizer = Quantizer(bits=8, granularity="per_tensor")
    path = tmp_path / "int8.pt"

    quantize_and_save(x, quantizer, path)
    payload, _, metadata = load_quantized(path)

    assert payload.dtype == torch.int8
    assert payload.shape == x.shape
    assert metadata["packing"] is None
