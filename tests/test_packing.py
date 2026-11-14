
import torch
import pytest

from quant_engine.packing import (
    pack_int4,
    unpack_int4
)


def test_basic_packing():

    x = torch.tensor([-2, 3], dtype=torch.int8)

    packed, metadata = pack_int4(x)

    assert packed.dtype == torch.uint8
    assert packed.numel() == 1

    assert packed.item() == 0x3E

    reconstructed = unpack_int4(packed, metadata)

    assert torch.equal(x, reconstructed)


def test_negative_values():

    x = torch.tensor(
        [-8, -1, 0, 7],
        dtype=torch.int8
    )

    packed, metadata = pack_int4(x)

    reconstructed = unpack_int4(packed, metadata)

    assert torch.equal(x, reconstructed)


def test_odd_number_of_values():

    x = torch.tensor(
        [1, -2, 3, -4, 5],
        dtype=torch.int8
    )

    packed, metadata = pack_int4(x)

    reconstructed = unpack_int4(packed, metadata)

    assert torch.equal(x, reconstructed)

    assert packed.numel() == 3


def test_tensor_shape_preservation():

    x = torch.tensor(
        [[1, 2], [-3, -4]],
        dtype=torch.int8
    )

    packed, metadata = pack_int4(x)

    reconstructed = unpack_int4(packed, metadata)

    assert reconstructed.shape == x.shape
    assert torch.equal(x, reconstructed)


def test_invalid_int4_values():

    x = torch.tensor(
        [-9, 2, 3, 4],
        dtype=torch.int8
    )

    with pytest.raises(ValueError):
        pack_int4(x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
