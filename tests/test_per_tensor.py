
import torch
import pytest

from quant_engine.per_tensor import (
    quantize_tensor,
    dequantize_tensor
)
from quant_engine.per_channel import (quantize_per_channel, dequantize_per_channel)


def test_per_tensor_quantization():

    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    q_tensor, scale = quantize_tensor(x)

    reconstructed = dequantize_tensor(
        q_tensor,
        scale
    )

    assert q_tensor.dtype == torch.int8
    assert reconstructed.shape == x.shape

    print("Original:", x)
    print("Quantized:", q_tensor)
    print("Scale:", scale)
    print("Reconstructed:", reconstructed)


def test_zero_tensor():
    x = torch.zeros(5)

    q, scale = quantize_tensor(x)
    reconstructed = dequantize_tensor(q, scale)

    assert torch.all(q == 0)
    assert torch.all(torch.isfinite(reconstructed))


def test_int8_range():
    x = torch.randn(1000) * 10

    q, scale = quantize_tensor(x)

    assert q.min() >= -128
    assert q.max() <= 127


def test_reconstruction_error():
    x = torch.randn(1000)

    q, scale = quantize_tensor(x)
    reconstructed = dequantize_tensor(q, scale)

    error = torch.mean((x - reconstructed) ** 2)

    assert error < 0.001


def test_quantization_dtype():
    x = torch.randn(100)

    q, scale = quantize_tensor(x)

    assert q.dtype == torch.int8
    assert scale.dtype == x.dtype


def test_fp16_quantization():
    x = torch.randn(4, 4, dtype=torch.float16)

    q, scale = quantize_tensor(x)

    assert q.dtype == torch.int8
    assert torch.isfinite(scale).all()


def test_empty_tensor():
    x = torch.tensor([])

    with pytest.raises(ValueError):
        quantize_tensor(x)


def test_invalid_axis():
    x = torch.randn(4, 4)

    with pytest.raises(ValueError):
        quantize_per_channel(x, axis=5)


def test_cuda_compatibility():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    x = torch.randn(100, device="cuda")

    q, scale = quantize_tensor(x)

    assert q.device.type == "cuda"
    assert scale.device.type == "cuda"

if __name__ == "__main__":
    test_per_tensor_quantization()
