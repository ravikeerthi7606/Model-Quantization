
import torch

from quant_engine.per_group import (
    quantize_per_group,
    dequantize_per_group
)


def test_per_group_quantization():

    x = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [1.0, 2.0, 3.0, 4.0]
    ])

    q, scale = quantize_per_group(x, group_size=2)

    reconstructed = dequantize_per_group(
        q, scale, group_size=2
    )

    assert q.dtype == torch.int8
    assert q.shape == x.shape
    assert scale.shape == (2, 2, 1)

    assert torch.allclose(
        reconstructed,
        x,
        atol=scale.max().item() / 2 + 1e-7
    )


def test_non_divisible_group():

    x = torch.randn(3, 7)

    q, scale = quantize_per_group(x, group_size=4)

    reconstructed = dequantize_per_group(
        q, scale, group_size=4
    )

    assert q.shape == x.shape
    assert reconstructed.shape == x.shape
    assert torch.isfinite(reconstructed).all()


def test_zero_tensor():

    x = torch.zeros(2, 8)

    q, scale = quantize_per_group(x, group_size=4)

    assert torch.all(q == 0)
    assert torch.isfinite(scale).all()


if __name__ == "__main__":
    test_per_group_quantization()
    test_non_divisible_group()
    test_zero_tensor()

    print("All per-group tests passed!")
