
import torch

from .packing import pack_int4, unpack_int4


def compress_int4(tensor, quantizer):

    if quantizer.bits != 4:
        raise ValueError("compress_int4 requires a 4-bit Quantizer")

    # Step 1: Quantization
    q_tensor, quant_params = quantizer.quantize(tensor)

    # Step 2: Pack INT4 values
    packed_tensor, packing_params = pack_int4(q_tensor)

    # Step 3: Store metadata
    metadata = {
        "quantization": quant_params,
        "packing": packing_params,
        "original_dtype": str(tensor.dtype)
    }

    return packed_tensor, metadata


def decompress_int4(packed_tensor, metadata, quantizer):

    if quantizer.bits != 4:
        raise ValueError("decompress_int4 requires a 4-bit Quantizer")

    # Step 1: Unpack INT4 values
    q_tensor = unpack_int4(
        packed_tensor,
        metadata["packing"]
    )

    # Step 2: Dequantize
    reconstructed = quantizer.dequantize(
        q_tensor,
        metadata["quantization"]
    )

    # Step 3: Restore original floating-point dtype
    dtype_name = metadata["original_dtype"]
    original_dtype = getattr(torch, dtype_name.split(".")[-1])

    return reconstructed.to(original_dtype)
