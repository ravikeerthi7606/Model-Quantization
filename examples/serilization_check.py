import torch
from quant_engine.serialization import (
    save_quantized,
    load_quantized,
)

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

save_quantized(
    payload,
    scales,
    metadata,
    "checkpoints/quantized_model.pt"
)

q, s, meta = load_quantized(
    "checkpoints/quantized_model.pt"
)

print(q)
print(s)
print(meta)