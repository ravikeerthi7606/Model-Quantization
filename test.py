import numpy as np

weights = np.array([
    0.1, 0.2, 0.3, 0.4,
    2.0, 3.0, 4.0, 5.0
])

print(np.array_split(weights,2))