# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE_PATH = "/home/zhangchi/workarea/code/ai_study/code/data/boston.csv"
ALPHA = 0.001
ITER_TOTAL = 100
THRESHOLD = 150


class LinearRegressionMachine:
    def __init__(
        self,
        data: np.ndarray,
        alpha: float,
        feature_num: int,
        iter_total: int,
        threshold: float,
    ) -> None:
        # Examples is the input of X, where x_0 = 1
        self.__examples: np.ndarray = np.hstack((np.ones((data.shape[0], 1)), data))
        self.__alpha = alpha  # Learning rate
        self.__dim = self.__examples.shape[1] - 1  # Num of columns in X
        self.__feature_num = feature_num if feature_num <= self.__dim else self.__dim
        self.__iter_total = iter_total
        # The parameters of hypotheses, i.e. h(x) = \theta^T * x
        self.__theta = np.random.rand(self.__feature_num + 1)
        self.__examples_total: int = self.__examples.shape[0]
        self.__threshold = threshold

    def set_theta(self, theta: np.ndarray) -> None:
        if theta.shape != self.__theta.shape:
            raise Exception(
                "The shape of input ({}) is not equal to theta ({})".format(
                    theta.shape, self.__theta.shape
                )
            )

        self.__theta = theta

    def hypotheses(self, x: np.ndarray[float]) -> float:
        return np.dot(self.__theta, x)

    def loss_func(self) -> float:
        sum = 0.0

        for i in range(self.__examples_total):
            x_i = self.__examples[i, :][: self.__feature_num + 1]
            y_i = self.__examples[i, :][self.__dim]
            sum += pow((self.hypotheses(x_i) - y_i), 2)

        return 0.5 * sum

    def partial_derivative_theta(self, j: int, example: np.ndarray[float]) -> float:
        end = example.size - 1

        x = example[: self.__feature_num + 1]
        y = example[end]
        x_j = x[j]

        return (self.hypotheses(x) - y) * x_j

    def run(self) -> np.ndarray:
        res = np.empty((self.__iter_total + 1, self.__feature_num + 2))

        for i in range(self.__theta.size):
            res[0, i] = self.__theta[i]

        loss = self.loss_func()

        res[0, self.__theta.size] = loss

        for k in range(self.__iter_total):
            if loss <= self.__threshold:
                break

            for j in range(self.__theta.size):
                pd = 0.0
                for i in range(self.__examples_total):
                    x_i = self.__examples[i, :]
                    pd += self.partial_derivative_theta(j, x_i)

                self.__theta[j] -= self.__alpha * pd
                res[k + 1, j] = self.__theta[j]

            loss = self.loss_func()
            res[k + 1, self.__theta.size] = loss

        return res


def draw_loss_func(m: LinearRegressionMachine):
    begin = -100
    end = 100
    total = 101

    X, Y = np.meshgrid(np.linspace(begin, end, total), np.linspace(begin, end, total))

    Z = np.empty(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            m.set_theta(np.array([X[i, j], Y[i, j]]))
            Z[i, j] = m.loss_func()

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X, Y, Z, s=10)
    # ax.plot_surface(X, Y, Z)

    m.set_theta(np.array([100.0, 100.0]))
    res = m.run()
    ax.plot(res[:, 0], res[:, 1], res[:, 2], c="red")
    ax.scatter(res[:, 0], res[:, 1], res[:, 2], s=40, c="red")

    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    ax.set_zlabel("Loss Function")

    # plt.contour(X, Y, Z)
    # plt.colorbar()

    # plt.savefig("loss_func.png", dpi=600, format="png")

    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    data = df.values
    data = data[:20, :]
    data[:, 0] = data[:, 5].copy()

    m = LinearRegressionMachine(
        data=data,
        alpha=ALPHA,
        feature_num=1,
        iter_total=ITER_TOTAL,
        threshold=THRESHOLD,
    )

    # point_size = 2
    # x = m.__examples[:, 1]
    # y = m.__examples[:, m.__dim]

    # plt.scatter(x, y, point_size)
    # plt.xlabel("RM")
    # plt.ylabel("MDEV")
    # plt.grid(True)

    # m.run()
    # print(m.__theta, m.loss_func())
    # x = np.linspace(5, 8, 10)
    # y = np.array([m.__theta[0] + m.__theta[1] * e for e in x])
    # plt.scatter(x, y, point_size)
    # plt.plot(x, y)

    # plt.savefig("test.png", dpi=600, format="png")

    draw_loss_func(m)
