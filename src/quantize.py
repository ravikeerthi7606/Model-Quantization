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


def tensor_per_dequantize(quantize, scale, original):

    dequantize = quantize*scale

    mse = np.mean((tensor-dequantize)**2)

    return dequantize, mse


tensor = np.array([-456,-32,0,1,4,23,7,34,67,500],dtype=np.float32)
quantize,scale = tensor_per_quantize(tensor)
dequantize, mse = tensor_per_dequantize(quantize, scale, tensor)

print(f'Tensor(FLOAT32):{tensor}')
print(f'Scale:{scale}\nQuantization : {quantize}')
print(f'Dequantization : {dequantize}')
print(f'Reconstruction Error : {mse}')