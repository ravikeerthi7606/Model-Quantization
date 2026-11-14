
import torch
import pytest

from quant_engine import Quantizer


@pytest.mark.parametrize(
    "granularity",
    ["per_tensor", "per_channel", "per_group"]
)
def test_quantizer(granularity):

    x = torch.randn(4, 8)

    quantizer = Quantizer(
        bits=8,
        scheme="symmetric",
        granularity=granularity,
        group_size=4,
        axis=0 if granularity == "per_channel" else -1
    )

    q, params = quantizer.quantize(x)

    reconstructed = quantizer.dequantize(q, params)

    assert q.dtype == torch.int8
    assert reconstructed.shape == x.shape
    assert torch.isfinite(reconstructed).all()

    print(f"\nGranularity: {granularity}")
    print("Quantized shape:", q.shape)
    print("Scale shape:", params["scale"].shape)


def test_invalid_granularity():

    with pytest.raises(ValueError):
        Quantizer(granularity="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
