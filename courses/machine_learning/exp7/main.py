import numpy as np
from numpy import *
import imageio
import matplotlib.pyplot as plt

def zero_center(data):
    mean = np.mean(data, axis=0)
    return data - mean

def pca(data, k):
    data = zero_center(data)
    value, vector = np.linalg.eig(np.matmul(data, data.T))
    index = np.argsort(value)[-k:]
    index = list(reversed(index))
    vector = vector[:, index]
    vector = np.matmul(data.T, vector)
    # vector = vector / np.linalg.norm(vector, axis=0)
    return vector


def load_data():
    k = 4  # 0-10
    r = random.permutation(10) + 1
    data_train = []
    label_train = []
    data_test = []
    label_test = []
    for i in range(1, 41):
        for j in range(0, k):
            data = imageio.imread(f'orl_faces/s{i}/{r[j]}.pgm').reshape([-1])
            data_train.append(data)
            label_train.append(i)
        for j in range(k, 10):
            data = imageio.imread(f'orl_faces/s{i}/{r[j]}.pgm').reshape([-1])
            data_test.append(data)
            label_test.append(i)
    data_train = np.stack(data_train).astype(np.float32)
    label_train = np.stack(label_train).astype(np.float32)
    data_test = np.stack(data_test).astype(np.float32)
    label_test = np.stack(label_test).astype(np.float32)
    return data_train, label_train, data_test, label_test


def main():
    data_train, label_train, data_test, label_test = load_data()
    print("data_train: ", data_train.shape)
    print("label_train: ", label_train.shape)
    print("data_test: ", data_test.shape)
    print("label_test: ", label_test.shape)
    vector = pca(data_train, 40)
    print("vector: ", vector.shape)
    data_test = np.matmul(data_test - np.mean(data_train, axis=0), vector)
    data_train = np.matmul(zero_center(data_train), vector)
    cnt = 0
    for i in range(data_test.shape[0]):
        dis = np.sum(np.power(data_train - data_test[i], 2), axis=1)
        if label_train[np.argmin(dis)] == label_test[i]:
            cnt += 1
    print(f"%2d%%" % (cnt / data_test.shape[0] * 100))


if __name__ == '__main__':
    main()
