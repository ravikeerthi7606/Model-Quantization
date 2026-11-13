
import numpy as np
import matplotlib.pyplot as plt

tensor = np.array(
    [-456, -32, 0, 1, 4, 23, 7, 34, 67, 500],
    dtype=np.float32
)

scales = np.linspace(0.1, 5.0, 100)

errors = []

for scale in scales:

    quantized = np.round(tensor / scale)
    quantized = np.clip(quantized, -128, 127)

    dequantized = quantized * scale

    mse = np.mean((tensor - dequantized) ** 2)

    errors.append(mse)

plt.figure(figsize=(10, 5))
plt.plot(scales, errors)
plt.xlabel("Scale")
plt.ylabel("Reconstruction MSE")
plt.title("Scale vs Quantization Error")
plt.grid()
plt.show()

best_index = np.argmin(errors)

print("Best scale:", scales[best_index])
print("Minimum MSE:", errors[best_index])
