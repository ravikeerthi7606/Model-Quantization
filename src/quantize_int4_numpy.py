import numpy as np

def quantize_int4(tensor):
    q_min = -8
    q_max = 7

    scale = np.max(np.abs(tensor))/q_max

    quantize = np.round(tensor/scale)
    quantize = np.clip(quantize, q_min, q_max)
    return quantize.astype('int8'),scale


def pack_int4(values):
    values = np.asarray(values, dtype=np.int8)

    if np.any(values > 7) or np.any(-8 > values):
        raise ValueError("Values should be with the range [-8,7]")

    unsigned = values.astype(np.uint8) & 0x0f

    # Handle odd number of values by padding with zero
    if len(unsigned) % 2 != 0:
        unsigned = np.append(unsigned, np.uint8(0))

    high = unsigned[0::2]
    low = unsigned[1::2]

    packed = (high<<4) | low

    return packed.astype(np.uint8)


def unpack_int4(packed_values, original_length=None):

    packed_values = np.asarray(packed_values, dtype=np.uint8)

    high = (packed_values >> 4) & 0x0f
    low = packed_values & 0x0f

    values = np.empty(len(packed_values)*2, dtype=np.int8)

    values[0::2] = high
    values[1::2] = low

    values = np.where(values>=8, values-16, values)

    if original_length is not None:
        values = values[:original_length]

    return values.astype(np.int8)


def dequantize_int4(tensor, scales, original):

    dequantized = tensor*scales

    mse = np.mean((original-dequantized)**2)

    return dequantized, mse

# weights = np.array([-4.5, -2.3, -1.0, 0.0, 0.5, 1.2, 2.5, 4.0],dtype=np.float32)
# # weights = np.array([23,44,-1,4,-6,7,2,8,6],dtype=np.float32)
# values, scales = quantize_int4(weights)
# packed = pack_int4(values)
# unpacked_values = unpack_int4(packed,8)
# dequantized, mse = dequantize_int4(unpacked_values, scales, weights)

# print(f'Weights : {weights}')
# print(f'Quantized Values : {values}')
# print(f'Scales : {scales}')
# print(f'packed values : {packed}')
# print(f'Unpacked values : {unpacked_values}')
# print(f'Dequantized values : {dequantized}\nReconstruction error : {mse}')

values = np.array([-3, 2, 5, -1, 7], dtype=np.int8)

packed = pack_int4(values)
unpacked = unpack_int4(packed, len(values))

print("Original:", values)
print("Packed:", packed)
print("Unpacked:", unpacked)

print("Recovery:", np.array_equal(values, unpacked))
print("INT8 storage:", values.nbytes)
print("Packed INT4 storage:", packed.nbytes)