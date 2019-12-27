import os
import operator
from numpy import *
import matplotlib.pyplot as plt
import imageio


# define PCA
def pca(data, k):
    data = float32(mat(data))
    print("--------",data.shape)
    rows, cols = data.shape  # 取大小
    data_mean = mean(data, 0)
    data_mean_all = tile(data_mean, (rows, 1))
    Z = data - data_mean_all  # 中心化
    print("Z",Z.shape)
    T1 = Z * Z.T  # 计算样本的协方差
    print("T1",T1.shape)
    D, V = linalg.eig(T1)  # 特征值与特征向量
    V1 = V[:, 0:k]  # 取前k个特征向量
    V1 = Z.T * V1
    for i in range(k):  # 特征向量归一化
        L = linalg.norm(V1[:, i])
        V1[:, i] = V1[:, i] / L

    data_new = Z * V1  # 降维后的数据
    return data_new, data_mean, V1  # 训练结果


def load_data(k):
    print("--Getting data set--- ")
    dataSetDir = 'orl_faces'
    # 读取文件夹
    choose = random.permutation(10) + 1  # 随机排序1-10 (0-9）+1
    train_face = zeros((40 * k, 112 * 92))
    train_face_number = zeros(40 * k)
    test_face = zeros((40 * (10 - k), 112 * 92))
    test_face_number = zeros(40 * (10 - k))
    for i in range(40):  # 40 sample people
        people_num = i + 1
        for j in range(10):  # everyone has 10 different face
            if j < k:
                filename = dataSetDir + '/s' + str(people_num) + '/' + str(choose[j]) + '.pgm'
                img = imageio.imread(filename).reshape([1, -1])
                train_face[i * k + j, :] = img
                train_face_number[i * k + j] = people_num
            else:
                filename = dataSetDir + '/s' + str(people_num) + '/' + str(choose[j]) + '.pgm'
                img = imageio.imread(filename).reshape([1, -1])
                test_face[i * (10 - k) + (j - k), :] = img
                test_face_number[i * (10 - k) + (j - k)] = people_num

    return train_face, train_face_number, test_face, test_face_number


def main():
    train_face, train_face_number, test_face, test_face_number = load_data(4)
    print(train_face.shape, train_face_number.shape, test_face.shape, test_face_number.shape)
    data_train_new, data_mean, V = pca(train_face, 40)
    print(data_train_new.shape, data_mean.shape, V.shape)

    num_train = data_train_new.shape[0]
    num_test = test_face.shape[0]
    print("tile(data_mean, (num_test, 1)):", tile(data_mean, (num_test, 1)).shape)
    temp_face = test_face - tile(data_mean, (num_test, 1))
    data_test_new = temp_face * V  # 对测试集进行降维
    data_test_new = array(data_test_new)  # mat change to array
    data_train_new = array(data_train_new)
    true_num = 0
    for i in range(num_test):
        testFace = data_test_new[i, :]
        diffMat = data_train_new - tile(testFace, (num_train, 1))
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        sortedDistIndicies = sqDistances.argsort()
        indexMin = sortedDistIndicies[0]
        if train_face_number[indexMin] == test_face_number[i]:
            true_num += 1

    accuracy = float(true_num) / num_test
    print('The classify accuracy is: %.2f%%' % (accuracy * 100))


if __name__ == '__main__':
    main()
