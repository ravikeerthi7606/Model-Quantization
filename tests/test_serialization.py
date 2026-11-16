import pytest
import torch

from quant_engine.serialization import save_quantized, load_quantized


@pytest.fixture
def sample():
    payload = torch.tensor([32, 64, 95, 127], dtype=torch.int8)
    scales = torch.tensor([0.037795])
    metadata = {
        "bits": 8,
        "scheme": "symmetric",
        "granularity": "per_tensor",
        "original_shape": (2, 2),
        "original_numel": 4,
        "original_dtype": "torch.float32",
        "axis": None,
        "group_size": None,
        "packing": None,
    }
    return payload, scales, metadata


def test_save_load_payload(tmp_path, sample):
    payload, scales, metadata = sample
    path = tmp_path / "quantized.pt"
    save_quantized(payload, scales, metadata, path)
    loaded, _, _ = load_quantized(path)
    assert torch.equal(payload, loaded)


def test_save_load_scales(tmp_path, sample):
    payload, scales, metadata = sample
    path = tmp_path / "quantized.pt"
    save_quantized(payload, scales, metadata, path)
    _, loaded, _ = load_quantized(path)
    assert torch.equal(scales, loaded)


def test_save_load_metadata(tmp_path, sample):
    payload, scales, metadata = sample
    path = tmp_path / "quantized.pt"
    save_quantized(payload, scales, metadata, path)
    _, _, loaded = load_quantized(path)
    assert metadata == loaded


def test_missing_metadata_field(tmp_path, sample):
    payload, scales, metadata = sample
    del metadata["bits"]
    with pytest.raises(ValueError):
        save_quantized(payload, scales, metadata, tmp_path / "bad.pt")


def test_incorrect_numel(tmp_path, sample):
    payload, scales, metadata = sample
    metadata["original_numel"] = 5
    with pytest.raises(ValueError):
        save_quantized(payload, scales, metadata, tmp_path / "bad.pt")


def test_odd_int4_payload(tmp_path):
    payload = torch.tensor([0x2D, 0xF5, 0x07], dtype=torch.uint8)
    scales = torch.tensor([0.5])
    metadata = {
        "bits": 4,
        "scheme": "symmetric",
        "granularity": "per_tensor",
        "original_shape": (5,),
        "original_numel": 5,
        "original_dtype": "torch.float32",
        "axis": None,
        "group_size": None,
        "packing": {"original_shape": (5,), "original_numel": 5},
    }
    path = tmp_path / "odd.pt"
    save_quantized(payload, scales, metadata, path)
    loaded, loaded_scales, loaded_metadata = load_quantized(path)
    assert torch.equal(payload, loaded)
    assert torch.equal(scales, loaded_scales)
    assert loaded_metadata["original_numel"] == 5
