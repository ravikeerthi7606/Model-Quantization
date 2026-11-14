
import torch
import pytest

from quant_engine.benchmark import benchmark_quantization


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize(
    "granularity",
    ["per_tensor", "per_channel", "per_group"]
)
def test_benchmark(bits, granularity):

    x = torch.randn(8, 16)

    result = benchmark_quantization(
        x,
        bits=bits,
        granularity=granularity,
        group_size=4,
        axis=0 if granularity == "per_channel" else -1,
        warmup=1,
        iterations=3
    )

    assert result["original_bytes"] == x.numel() * 4

    assert result["compressed_bytes"] > 0

    assert result["quantize_ms"] >= 0
    assert result["dequantize_ms"] >= 0

    assert result["compression_ratio"] > 0

    assert "mse" in result["metrics"]
    assert "sqnr_db" in result["metrics"]


def test_int4_packed_storage():

    x = torch.randn(1000)

    result = benchmark_quantization(
        x,
        bits=4,
        granularity="per_tensor",
        warmup=1,
        iterations=3
    )

    assert result["payload_bytes"] == 500


def test_int8_storage():

    x = torch.randn(1000)

    result = benchmark_quantization(
        x,
        bits=8,
        granularity="per_tensor",
        warmup=1,
        iterations=3
    )

    assert result["payload_bytes"] == 1000


def test_invalid_iterations():

    x = torch.randn(100)

    with pytest.raises(ValueError):

        benchmark_quantization(
            x,
            iterations=0
        )
