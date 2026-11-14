
import torch


def pack_int4(q_tensor: torch.Tensor):

    if not isinstance(q_tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    if q_tensor.dtype != torch.int8:
        raise TypeError("Input must have torch.int8 dtype")

    if q_tensor.numel() == 0:
        raise ValueError("Input tensor cannot be empty")

    if torch.any(q_tensor < -8) or torch.any(q_tensor > 7):
        raise ValueError("Values must be within INT4 range [-8, 7]")

    original_shape = tuple(q_tensor.shape)
    original_numel = q_tensor.numel()

    # Flatten tensor
    x = q_tensor.reshape(-1).to(torch.int16)

    # Pad if odd number of values
    if original_numel % 2 != 0:
        x = torch.cat([
            x,
            torch.zeros(1, dtype=torch.int16, device=x.device)
        ])

    # Convert signed values to 4-bit representation
    x = x & 0x0F

    # Separate low and high nibbles
    low = x[0::2]
    high = x[1::2]

    # Combine two nibbles into one byte
    packed = low | (high << 4)

    return packed.to(torch.uint8), {
        "original_shape": original_shape,
        "original_numel": original_numel
    }


def unpack_int4(packed, metadata):

    if packed.dtype != torch.uint8:
        raise TypeError("Packed tensor must have torch.uint8 dtype")

    original_numel = metadata["original_numel"]
    original_shape = metadata["original_shape"]

    if packed.numel() != (original_numel + 1) // 2:
        raise ValueError("Packed data size does not match metadata")

    # Extract nibbles
    packed = packed.to(torch.int16)

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    # Interleave nibbles
    values = torch.stack([low, high], dim=1).reshape(-1)

    # Convert unsigned nibble to signed INT4
    values = torch.where(
        values >= 8,
        values - 16,
        values
    )

    # Remove padding
    values = values[:original_numel]

    return values.to(torch.int8).reshape(original_shape)
