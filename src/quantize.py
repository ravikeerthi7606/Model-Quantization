import numpy as np

# per tensor quantization
def tensor_per_quantize(tensor):

    #INT8 quantization
    q_min = -128
    q_max = 127

    scale = np.max(np.abs(tensor))/q_max

    quantize = np.round(tensor/scale)
    quantize = np.clip(quantize, q_min, q_max)

    return quantize.astype(np.int8), scale


def tensor_per_dequantize(quantized, scale, tensor):

    dequantize = quantized*scale

    mse = np.mean((tensor-dequantize)**2)

    return dequantize, mse


tensor = np.array([-456,-32,0,1,4,23,7,34,67,500],dtype=np.float32)
quantize,scale = tensor_per_quantize(tensor)
dequantize, mse = tensor_per_dequantize(quantize, scale, tensor)

print(f'Tensor(FLOAT32):{tensor}')
print(f'Scale:{scale}\nQuantization : {quantize}')
print(f'Dequantization : {dequantize}')
print(f'Reconstruction Error : {mse}')

#per channel Quantization
def per_channel_quantize(tensor):
    #zero point for INT8
    q_min = -128
    q_max = 127

    scale = np.max(np.abs(tensor), axis=1, keepdims=True)/q_max
    quantize = np.round(tensor/scale)
    quantize = np.clip(quantize, q_min, q_max)
    return quantize.astype('int8'),scale

def per_channel_dequantize(quantized, scale, tensor):
    dequantize = quantized*scale
    mse = np.mean((tensor-dequantize)**2)
    return dequantize, mse

weights = np.array([
    [0.1, 0.2, 0.3],
    [2.0, 3.0, 4.0]
], dtype=np.float32)

# Per-tensor
qt, st = tensor_per_quantize(weights)
dt, mse_t = tensor_per_dequantize(qt, st, weights)

# Per-channel
qc, sc = per_channel_quantize(weights)
dc, mse_c = per_channel_dequantize(qc, sc, weights)


print("Original weights:\n", weights)

print("\nPer-tensor quantized:\n", qt)
print("Scale:", st)
print("MSE:", mse_t)

print("\nPer-channel quantized:\n", qc)
print("Scales:\n", sc)
print("MSE:", mse_c)
