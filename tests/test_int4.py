
import torch
import pytest

from quant_engine import Quantizer


@pytest.mark.parametrize(
    "granularity",
    ["per_tensor", "per_channel", "per_group"]
)
@pytest.mark.parametrize(
    "bits",
    [4, 8]
)
def test_all_quantization_modes(granularity, bits):

    x = torch.randn(8, 16)

    quantizer = Quantizer(
        bits=bits,
        scheme="symmetric",
        granularity=granularity,
        group_size=4,
        axis=0 if granularity == "per_channel" else -1
    )

    q, params = quantizer.quantize(x)

    reconstructed = quantizer.dequantize(q, params)

    q_min = -(2 ** (bits - 1))
    q_max = (2 ** (bits - 1)) - 1

    assert q.dtype == torch.int8
    assert q.min() >= q_min
    assert q.max() <= q_max
    assert reconstructed.shape == x.shape
    assert torch.isfinite(reconstructed).all()


def test_int4_zero_tensor():

    x = torch.zeros(4, 8)

    quantizer = Quantizer(
        bits=4,
        granularity="per_group",
        group_size=4
    )

    q, params = quantizer.quantize(x)

    assert torch.all(q == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
