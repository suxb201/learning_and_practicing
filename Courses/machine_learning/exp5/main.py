import numpy as np
import matplotlib.pyplot as plt
import cvxopt
import cvxopt.solvers


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


class SVM:
    def __init__(self, kernel_func, term_c):
        self.kernel_func = kernel_func
        self.c = c

        self.a = None
        self.sv = None
        self.sv_y = None
        self.b = None
        self.w = None

    def fit(self, x, y):
        n, d = x.shape()  # d=2
        k = np.array([[self.kernel_func(x[i], x[j]) for j in range(n)] for i in range(n)])
        p = cvxopt.matrix(np.outer(y, y) * k)
        q = cvxopt.matrix(-1 * np.ones(n))
        a = cvxopt.matrix(y, (1, n))
        b = cvxopt.matrix(.0)
        g = cvxopt.matrix(np.vstack([np.diag(np.ones(n) * -1), np.identity(n)]))
        h = cvxopt.matrix(np.hstack([np.zeros(n), np.ones(n) * self.c]))

        solution = cvxopt.solvers.qp(p, q, g, h, a, b)

        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, x):
        if self.w is not None:
            return np.dot(x, self.w) + self.b
        else:
            y_predict = np.zeros(len(x))
            for i in range(len(x)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, x):
        return np.sign(self.project(x))


def train(path_train, path_test):
    data_train = np.fromstring(open(path_train).read(), dtype=np.float32, sep=' ').reshape([-1, 3])
    data_test = np.fromstring(open(path_test).read(), dtype=np.float32, sep=' ').reshape([-1, 3])

    x = data_train[:, :2]
    y = data_train[:, 2]

    plt.figure()
    plt.title(path_train)
    plt.scatter(x[:, 0].reshape(-1), x[:, 1].reshape(-1), s=23, c=y.reshape(-1), cmap=plt.get_cmap('Dark2'), alpha=0.79)
    plt.show()


def test():
    pass


if __name__ == '__main__':
    train('.ignore/training_1.txt', '.ignore/test_1.txt')
    train('.ignore/training_2.txt', '.ignore/test_2.txt')
