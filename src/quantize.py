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


def tensor_per_dequantize(quantize, scale):

    dequantize = quantize*scale

    

    return dequantize


tensor = np.array([-456,-32,0,1,4,23,7,34,67,500])
quantize,scale = tensor_per_quantize(tensor)
dequantize = tensor_per_dequantize(quantize, scale)

print(f'Tensor(FLOAT32):{tensor}')
print(f'Scale:{scale}\nQuantization : {quantize}')
print(f'Dequantization : {dequantize}')