
import torch
import pytest

from quant_engine import Quantizer
from quant_engine.compression import (
    compress_int4,
    decompress_int4
)


@pytest.mark.parametrize(
    "granularity",
    ["per_tensor", "per_channel", "per_group"]
)
def test_int4_compression_pipeline(granularity):

    x = torch.randn(8, 16)

    quantizer = Quantizer(
        bits=4,
        scheme="symmetric",
        granularity=granularity,
        group_size=4,
        axis=0 if granularity == "per_channel" else -1
    )

    packed, metadata = compress_int4(x, quantizer)

    reconstructed = decompress_int4(
        packed,
        metadata,
        quantizer
    )

    assert packed.dtype == torch.uint8
    assert reconstructed.shape == x.shape
    assert reconstructed.dtype == x.dtype
    assert torch.isfinite(reconstructed).all()


def test_compression_size():

    x = torch.randn(1000)

    quantizer = Quantizer(
        bits=4,
        granularity="per_tensor"
    )

    packed, metadata = compress_int4(x, quantizer)

    assert packed.numel() == 500


def test_fp16_compression():

    x = torch.randn(100, dtype=torch.float16)

    quantizer = Quantizer(
        bits=4,
        granularity="per_tensor"
    )

    packed, metadata = compress_int4(x, quantizer)

    reconstructed = decompress_int4(
        packed,
        metadata,
        quantizer
    )

    assert reconstructed.dtype == torch.float16


def test_reject_int8_quantizer():

    quantizer = Quantizer(bits=8)

    x = torch.randn(10)

    with pytest.raises(ValueError):
        compress_int4(x, quantizer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
