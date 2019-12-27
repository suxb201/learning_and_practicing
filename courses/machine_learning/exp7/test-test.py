import numpy as np

vector = np.array([[1, 2], [1, 2], [1, 2.0]])
print(np.linalg.norm(vector, axis=0))
# vector = vector / np.linalg.norm(vector, axis=0)
# print(vector)

a = [2, 1]
print(vector / a)
