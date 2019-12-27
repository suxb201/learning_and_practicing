import numpy as np
from numpy import *
import imageio
import matplotlib.pyplot as plt
from sklearn import svm


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
    vector = vector / np.linalg.norm(vector, axis=0)
    return vector


def load_data():
    k = 6  # 0-10
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
    data = np.concatenate([data_train, data_test], axis=0)
    vector = pca(data, 40)
    data = np.matmul(zero_center(data), vector)
    data = data / np.linalg.norm(data, axis=0)
    data_train, data_test = data[:label_train.shape[0], :], data[label_train.shape[0]:, :]

    res = []
    for i in range(1, 41):
        clf = svm.SVC(gamma=0.001, C=100, probability=True)  # class
        label = [1 if x == i else -1 for x in label_train]
        clf.fit(data_train, label)  # training the svc model
        res.append(clf.predict_proba(data_test)[:, 1])
    res = np.stack(res)
    print(f"%2d%%" % (sum(np.argmax(res, axis=0) + 1 == label_test) / data_test.shape[0] * 100))


if __name__ == '__main__':
    main()
