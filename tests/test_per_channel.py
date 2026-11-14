
import torch

from quant_engine.per_channel import (
    quantize_per_channel,
    dequantize_per_channel
)


def test_per_channel_quantization():

    x = torch.tensor([
        [0.1, 0.2, 0.3],
        [1.0, 2.0, 3.0],
        [0.01, 0.02, 0.03]
    ])

    q, scale = quantize_per_channel(x, axis=0)

    reconstructed = dequantize_per_channel(q, scale)

    assert q.dtype == torch.int8
    assert scale.shape == (3, 1)
    assert reconstructed.shape == x.shape

    assert q.min() >= -128
    assert q.max() <= 127

    assert torch.allclose(
        reconstructed,
        x,
        atol=scale.max().item() / 2 + 1e-7
    )


def test_per_channel_scales():

    x = torch.tensor([
        [0.1, 0.2, 0.3],
        [1.0, 2.0, 3.0],
        [0.01, 0.02, 0.03]
    ])

    _, scale = quantize_per_channel(x, axis=0)

    expected = torch.tensor([
        [0.3 / 127],
        [3.0 / 127],
        [0.03 / 127]
    ])

    assert torch.allclose(scale, expected)


def test_zero_channel():

    x = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0]
    ])

    q, scale = quantize_per_channel(x, axis=0)

    assert torch.all(q[0] == 0)
    assert torch.all(torch.isfinite(scale))


if __name__ == "__main__":
    test_per_channel_quantization()
    test_per_channel_scales()
    test_zero_channel()

    print("All per-channel tests passed!")
