import torch

#----Per Tensor Quantization----
def quantize(tensor):
    q_min,q_max = -128,127

    scale = torch.max(torch.abs(tensor))/q_max
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantize = torch.round(tensor/scale)
    quantize = torch.clamp(quantize, q_min, q_max)
    return quantize.to(torch.int8), scale

def dequantize(quantized, scale, original):
    dequantized = quantized.to(original.dtype)*scale

    mse = torch.mean((original - dequantized)**2)

    return dequantized, mse

#----Per Channel Quantization----
def channel_quantize(tensor):
    q_min,q_max = -128,127

    scale = torch.max(torch.abs(tensor), dim=1, keepdim=True).values/q_max

    quantize = torch.round(tensor/scale)
    quantize = torch.clamp(quantize, q_min, q_max)

    return quantize.to(torch.int8), scale

#----Per Group Quantization----
def group_quantize(tensor, group_size=4):
    q_min,q_max = -128,127
    quantized_groups = []
    scales = []
    for i in range(0,tensor.numel(),group_size):
        group = tensor[i:i+group_size]
        scale = torch.max(torch.abs(group))/q_max
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        quantize = torch.round(group/scale)
        quantize = torch.clamp(quantize, q_min, q_max)
        quantized_groups.append(quantize.to(torch.int8))
        scales.append(scale)
    return quantized_groups, scales

def group_dequantize(tensors, scales, original):
    dequantized = []
    for tensor,scale in zip(tensors, scales):
        dequantize = tensor*scale
        dequantized.append(dequantize)
    dequantized = torch.cat((dequantized))
    mse = torch.mean((original-dequantized)**2)

    return dequantized, mse


tensor = torch.tensor([
    12., 34., 65.,
    45., 22., 98.,
    -2., 6., -90.,
    18., -40., 7.
])

quantized, scales = group_quantize(tensor, group_size=4)

dequantized, mse = group_dequantize(
    quantized, scales, tensor
)

print("Original:", tensor)
print("Quantized groups:", quantized)
print("Scales:", scales)
print("Dequantized:", dequantized)
print("MSE:", mse.item())
print("Original memory:", tensor.numel() * tensor.element_size())
print(
    "Quantized memory:",
    sum(group.numel() * group.element_size() for group in quantized)
)
# tensors = torch.tensor([12.0, 34.0, 65.0,
#                        45.0, 22.0, 98.0,
#                        -2.0, 6.0, -90.0,
#                        18.0, -40.0, 7.0])
# print(group_quantize(tensors, group_size=4))
# print(group_dequantize(*group_quantize(tensors, group_size=4), tensors))
# tensor = torch.randn(50)
# tensor = torch.tensor([0,0,0,0])
# tensor = torch.tensor([-2.5, -1.0, 0.0, 1.5, 2.5])
# tensor = torch.randn(1000, device="cuda")
# quantized, scale = quantize(tensor)
# dequantized, mse = dequantize(quantized, scale, tensor)
# print(quantized.dtype,scale)
# print(dequantized.dtype, mse)


# tensor = torch.randn(1000, device="cpu")
# tensor = torch.tensor([[12,34,65],
#                        [45,22,98],
#                        [-2,6,-90]])
# quantized, scale = channel_quantize(tensor)
# dequantized, mse = dequantize(quantized, scale, tensor)

# print("Original dtype:", tensor.dtype)
# print("Quantized dtype:", quantized.dtype)

# print("Original memory:", tensor.numel() * tensor.element_size())
# print("Quantized memory:", quantized.numel() * quantized.element_size())

# print("Original device:", tensor.device)
# print("Quantized device:", quantized.device)

# print("MSE:", mse.item())

# tensor = torch.tensor([
#     [12., 34., 65.],
#     [45., 22., 98.],
#     [-2., 6., -90.]
# ])

# # Per-tensor
# q_tensor, s_tensor = quantize(tensor)
# dq_tensor, mse_tensor = dequantize(q_tensor, s_tensor, tensor)

# # Per-channel
# q_channel, s_channel = channel_quantize(tensor)
# dq_channel, mse_channel = dequantize(q_channel, s_channel, tensor)

# print("Per-tensor MSE:", mse_tensor.item())
# print("Per-channel MSE:", mse_channel.item())