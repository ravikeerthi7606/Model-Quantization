import torch


def get_quantization_range(bits: int):
    if not isinstance(bits, int) or isinstance(bits, bool):
        raise TypeError("bits must be an integer")

    if bits < 2 or bits > 8:
        raise ValueError("bits must be between 2 and 8")

    q_min = -(2 ** (bits - 1))
    q_max = (2 ** (bits - 1)) - 1

    return q_min, q_max
