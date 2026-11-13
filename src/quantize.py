import numpy as np

# per tensor quantization
def tensor_per_quantize(tensor):
    s = np.max(np.abs(tensor))

    #INT8 quantization
    q_min = -128
    q_max = 127

    quantize = np.round(tensor/s)
    quantize = np.clip(quantize, q_min, q_max)

    return quantize.astype(np.int8), s
    # S=0.001

    # q_min = 0
    # q_max = 255

    # scaling = tensor/S
    # quantize = np.clip(scaling, min=q_min, max=q_max)
    # return quantize


tensor = [1,4,23,7,34,67]
print(tensor_per_quantize(tensor))