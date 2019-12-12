import numpy as np

a = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float16)
b = np.array([1, 2], dtype=np.float16)
print(a)
print(b)
a /= b[:, None]
print(a)
