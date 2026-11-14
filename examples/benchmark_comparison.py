
import torch

from quant_engine.benchmark import benchmark_quantization


# weights = torch.randn(1_000_000)

# results = []

# for bits in [8, 4]:

#     for granularity in [
#         "per_tensor",
#         "per_channel",
#         "per_group"
#     ]:

#         result = benchmark_quantization(
#             weights,
#             bits=bits,
#             granularity=granularity,
#             group_size=128,
#             axis=0,
#             warmup=5,
#             iterations=20
#         )

#         results.append(result)


print("\nQUANTIZATION BENCHMARK")
print("=" * 90)

# for result in results:

#     print(
#         f"\n{result['bits']}-bit | "
#         f"{result['granularity']}"
#     )

#     print(
#         f"Compressed memory: "
#         f"{result['compressed_bytes'] / 1024**2:.3f} MB"
#     )

#     print(
#         f"Memory reduction: "
#         f"{result['memory_reduction_pct']:.2f}%"
#     )

#     print(
#         f"Quantization: "
#         f"{result['quantize_ms']:.3f} ms"
#     )

#     print(
#         f"Dequantization: "
#         f"{result['dequantize_ms']:.3f} ms"
#     )

#     print(
#         f"MSE: {result['metrics']['mse']:.8f}"
#     )

#     print(
#         f"SQNR: {result['metrics']['sqnr_db']:.3f} dB"
#     )

weights = torch.randn(256, 1024)

for bits in [4, 8]:

    for group_size in [32, 64, 128, 256]:

        result = benchmark_quantization(
            weights,
            bits=bits,
            granularity="per_group",
            group_size=group_size,
            axis=-1,
            warmup=5,
            iterations=20
        )

        print(
            f"Bits: {bits} | "
            f"Group: {group_size} | "
            f"MSE: {result['metrics']['mse']:.8f} | "
            f"SQNR: {result['metrics']['sqnr_db']:.2f} dB | "
            f"Memory: {result['compressed_bytes'] / 1024**2:.3f} MB"
        )

