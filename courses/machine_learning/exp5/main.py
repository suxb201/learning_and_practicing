import numpy as np
import matplotlib.pyplot as plt
import cvxopt
import cvxopt.solvers


def linear_kernel(x1, x2):
    return np.matmul(x1, x2.T)
    # return np.dot(x1, x2)


# def gaussian_kernel(x, y, sigma=5.0):
#     return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

def RBF_kernel(x, y, the_lambda=1000):
    return np.exp(-the_lambda * np.linalg.norm(x - y) ** 2)


class SVM:
    def __init__(self, kernel_func, term_c):
        self.kernel_func = kernel_func
        self.c = term_c

        self.a = None
        self.sv = None
        self.sv_y = None
        self.b = None
        self.w = None

    def fit(self, x, y):
        n, d = x.shape  # d=2
        # k = np.array([[self.kernel_func(x[i], x[j]) for j in range(n)] for i in range(n)], dtype=np.float64)
        k = linear_kernel(x, x)
        p = cvxopt.matrix(np.outer(y, y) * k)
        q = cvxopt.matrix(-1 * np.ones(n))
        a = cvxopt.matrix(y, (1, n))
        b = cvxopt.matrix(.0)
        if self.c == 0:
            g = cvxopt.matrix(np.diag(np.ones(n) * -1))
            h = cvxopt.matrix(np.zeros(n))
        else:
            g = cvxopt.matrix(np.vstack([np.diag(np.ones(n) * -1), np.identity(n)]))
            h = cvxopt.matrix(np.hstack([np.zeros(n), np.ones(n) * self.c]))
        print('solution = cvxopt.solvers.qp(p, q, g, h, a, b)')
        solution = cvxopt.solvers.qp(p, q, g, h, a, b)

        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = x[sv]
        self.sv_y = y[sv]
        print(f"support vectors: {len(self.a)}")

        # Intercept
        self.b = 0
        for i in range(len(self.a)):
            self.b += self.sv_y[i]
            self.b -= np.sum(self.a * self.sv_y * k[ind[i], sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel_func == linear_kernel:
            self.w = np.zeros(d)
            for i in range(len(self.a)):
                self.w += self.a[i] * self.sv_y[i] * self.sv[i]
        else:
            self.w = None

    def predict(self, x):
        if self.w is not None:
            return np.dot(x, self.w) + self.b
        else:
            y_predict = np.zeros(len(x))
            for i in range(len(x)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel_func(x[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def plot_margin(self, x1, x2, title):
        def f(x, w, b, c):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (-w[0] * x - b + c) / w[1]

        plt.figure()
        plt.title(title)
        plt.plot(x1[:, 0], x1[:, 1], "b+")
        plt.plot(x2[:, 0], x2[:, 1], "ro")
        plt.scatter(self.sv[:, 0], self.sv[:, 1], s=100, c="g")

        x_min = min(np.min(x1[:, 0]), np.min(x2[:, 0]))
        x_max = max(np.max(x1[:, 0]), np.max(x2[:, 0]))
        a0 = x_min
        a1 = f(a0, self.w, self.b, 0)
        b0 = x_max
        b1 = f(b0, self.w, self.b, 0)
        plt.plot([a0, b0], [a1, b1], "k")

        a0 = x_min
        a1 = f(a0, self.w, self.b, 1)
        b0 = x_max
        b1 = f(b0, self.w, self.b, 1)
        plt.plot([a0, b0], [a1, b1], "k--")

        a0 = x_min
        a1 = f(a0, self.w, self.b, -1)
        b0 = x_max
        b1 = f(b0, self.w, self.b, -1)
        plt.plot([a0, b0], [a1, b1], "k--")

        plt.axis("tight")
        plt.show()

    def plot_contour(self, x1, x2, title=''):

        plt.figure()
        plt.title(title)
        plt.plot(x1[:, 0], x1[:, 1], "bo")
        plt.plot(x2[:, 0], x2[:, 1], "wo")
        plt.scatter(self.sv[:, 0], self.sv[:, 1], s=100, c="g")
        x_min = min(np.min(x1[:, 0]), np.min(x2[:, 0]))
        x_max = max(np.max(x1[:, 0]), np.max(x2[:, 0]))
        y_min = min(np.min(x1[:, 1]), np.min(x2[:, 1]))
        y_max = max(np.max(x1[:, 1]), np.max(x2[:, 1]))
        x_list, y_list = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        dots = np.array([[x, y] for x, y in zip(np.ravel(x_list), np.ravel(y_list))])

        z = self.predict(dots)
        plt.contour(x_list, y_list, z.reshape(x_list.shape), [0], colors='k')
        z = (z >= 1).astype(np.float) + (z >= -1).astype(np.float)
        z = z.reshape(x_list.shape)
        plt.contourf(x_list, y_list, z)
        plt.show()


def problem1(path_train, path_test, term_c):
    data_train = np.fromstring(open(path_train).read(), dtype=np.float64, sep=' ').reshape([-1, 3])
    data_test = np.fromstring(open(path_test).read(), dtype=np.float64, sep=' ').reshape([-1, 3])
    x = data_train[:, :2]
    y = data_train[:, 2].reshape(-1)
    svm = SVM(kernel_func=linear_kernel, term_c=term_c)
    svm.fit(x, y)
    predict = np.sign(svm.predict(x))
    cnt = np.sum(predict == y)
    svm.plot_margin(x[y == 1], x[y == -1], f"problem1({path_train, path_test}) train error: {1.0 * cnt / len(data_train)}")

    x = data_test[:, :2]
    y = data_test[:, 2].reshape(-1)
    predict = np.sign(svm.predict(x))
    cnt = np.sum(predict == y)
    svm.plot_margin(x[y == 1], x[y == -1], f"problem1({path_train, path_test}) test error : {1.0 * cnt / len(data_test)}")


def problem2(term_c):
    def data_loader(data_path):
        x = []
        y = []
        for line in open(data_path).readlines():
            line = line.strip().split(' ')
            y.append(float(line[0]))
            tmp = np.zeros(784)
            for item_line in line[1:]:
                item_line = item_line.split(':')
                tmp[int(item_line[0])] = float(item_line[1]) / 255
            x.append(tmp)
        x = np.stack(x)
        y = np.array(y)
        return x, y

    def draw(data, title='draw'):
        data = data.reshape(28, 28)
        plt.figure()
        plt.title(title)
        plt.imshow(data, cmap=plt.cm.gray)
        plt.show()

    def the_zip(data):
        index = [True if i % 100 == 0 else False for i in range(784)]
        return data[:, index]

    x, y = data_loader('.ignore/train-01-images.svm')
    x, y = x[:200], y[:200]
    svm = SVM(linear_kernel, term_c)
    print(the_zip(x).shape)
    print(y.shape)
    svm.fit(the_zip(x), y)
    predict = np.sign(svm.predict(the_zip(x)))
    cnt = np.sum(predict == y)
    # print(np.where(predict != y))
    # print(np.where(predict != y)[0][0])
    draw(x[np.where(predict != y)[0][0]])
    print(f"problem2(.ignore/train-01-images.svm) train({len(x)}) error: {1.0 * cnt / len(x)}")

    x, y = data_loader('.ignore/test-01-images.svm')
    predict = np.sign(svm.predict(the_zip(x)))
    cnt = np.sum(predict == y)
    print(f"problem2(.ignore/test-01-images.svm) test({len(x)}) error: {1.0 * cnt / len(x)}")


def problem3():
    data = np.fromstring(open('.ignore/training_3.text').read(), dtype=np.float64, sep=' ').reshape([-1, 3])
    x = data[:, :2]
    y = data[:, 2].reshape(-1)
    svm = SVM(kernel_func=RBF_kernel, term_c=0)
    svm.fit(x, y)
    predict = np.sign(svm.predict(x))
    cnt = np.sum(predict == y)
    svm.plot_contour(x[y == 1], x[y == -1], f"problem3 lambda=1 error: {1.0 * cnt / len(data)}")
    # svm.plot_contour(x[y == 1], x[y == -1], f"problem3")


if __name__ == '__main__':
    # problem1('.ignore/training_1.txt', '.ignore/test_1.txt', 0)
    # problem1('.ignore/training_2.txt', '.ignore/test_2.txt', 0)
    problem2(1)
    # problem3()
