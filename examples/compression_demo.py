
import torch

from quant_engine import Quantizer, compress_int4, decompress_int4


weights = torch.randn(1_000_000)

quantizer = Quantizer(
    bits=4,
    scheme="symmetric",
    granularity="per_group",
    group_size=128,
    axis=-1
)

# Compress
packed, metadata = compress_int4(weights, quantizer)

# Reconstruct
reconstructed = decompress_int4(
    packed,
    metadata,
    quantizer
)

# Memory calculations
original_bytes = weights.numel() * weights.element_size()
packed_bytes = packed.numel() * packed.element_size()

compression_ratio = original_bytes / packed_bytes
memory_reduction = (
    1 - packed_bytes / original_bytes
) * 100

mse = torch.mean((weights - reconstructed) ** 2)

print("Original memory:", original_bytes, "bytes")
print("Packed memory:", packed_bytes, "bytes")
print("Compression ratio:", compression_ratio)
print("Memory reduction:", memory_reduction, "%")
print("Reconstruction MSE:", mse.item())
print("Original shape:", weights.shape)
print("Reconstructed shape:", reconstructed.shape)
