
import pytest
import torch

from quant_engine.serialization import (
    save_quantized,
    load_quantized,
)


@pytest.fixture
def sample_artifact():
    payload = torch.tensor(
        [32, 64, 95, 127],
        dtype=torch.int8
    )

    scales = torch.tensor([0.037795])

    metadata = {
        "bits": 8,
        "scheme": "symmetric",
        "granularity": "per_tensor",
        "original_shape": (2, 2),
        "original_numel": 4,
        "original_dtype": "float32",
        "axis": None,
        "group_size": None,
        "packing": "none",
    }

    return payload, scales, metadata


def test_save_load_payload(tmp_path, sample_artifact):
    payload, scales, metadata = sample_artifact
    path = tmp_path / "quantized.pt"

    save_quantized(payload, scales, metadata, path)

    loaded_payload, _, _ = load_quantized(path)

    assert torch.equal(payload, loaded_payload)


def test_save_load_scales(tmp_path, sample_artifact):
    payload, scales, metadata = sample_artifact
    path = tmp_path / "quantized.pt"

    save_quantized(payload, scales, metadata, path)

    _, loaded_scales, _ = load_quantized(path)

    assert torch.equal(scales, loaded_scales)


def test_save_load_metadata(tmp_path, sample_artifact):
    payload, scales, metadata = sample_artifact
    path = tmp_path / "quantized.pt"

    save_quantized(payload, scales, metadata, path)

    _, _, loaded_metadata = load_quantized(path)

    assert metadata == loaded_metadata


def test_missing_metadata_field(tmp_path, sample_artifact):
    payload, scales, metadata = sample_artifact

    del metadata["bits"]

    with pytest.raises(ValueError):
        save_quantized(
            payload, scales, metadata,
            tmp_path / "invalid.pt"
        )


def test_incorrect_original_numel(tmp_path, sample_artifact):
    payload, scales, metadata = sample_artifact

    metadata["original_numel"] = 5

    with pytest.raises(ValueError):
        save_quantized(
            payload, scales, metadata,
            tmp_path / "invalid.pt"
        )


def test_int4_odd_element_payload(tmp_path):
    # Three packed bytes represent five INT4 values.
    payload = torch.tensor(
        [0x2D, 0xF5, 0x07],
        dtype=torch.uint8
    )

    scales = torch.tensor([0.5])

    metadata = {
        "bits": 4,
        "scheme": "symmetric",
        "granularity": "per_tensor",
        "original_shape": (5,),
        "original_numel": 5,
        "original_dtype": "float32",
        "axis": None,
        "group_size": None,
        "packing": "int4_packed",
    }

    path = tmp_path / "int4.pt"

    save_quantized(payload, scales, metadata, path)

    loaded_payload, loaded_scales, loaded_metadata = (
        load_quantized(path)
    )

    assert torch.equal(payload, loaded_payload)
    assert torch.equal(scales, loaded_scales)
    assert loaded_metadata["original_numel"] == 5
