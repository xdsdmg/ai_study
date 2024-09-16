# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE_PATH = "/home/zhangchi/workarea/code/ai_study/code/data/boston.csv"
ALPHA = 0.001
ITER_TOTAL = 10


class LinearRegressionMachine:
    def __init__(
        self, data_file_path: str, alpha: float, feature_num: int, iter_total: int
    ) -> None:
        df = pd.read_csv(data_file_path)
        data = df.values

        # Examples is the input of X, x_0 = 1
        self.examples: np.ndarray = np.hstack((np.ones((data.shape[0], 1)), data))
        self.examples = self.examples[:20, :]  # TODO: need op

        # TODO: tmp code
        self.examples[:, 1] = self.examples[:, 6].copy()

        self.alpha = alpha  # Learning rate
        self.dim = self.examples.shape[1] - 1  # Num of columns in X
        self.feature_num = feature_num if feature_num <= self.dim else self.dim
        self.iter_total = iter_total
        # The parameters of hypotheses, i.e. h(x) = \theta^T * x
        # self.theta = np.random.rand(self.feature_num + 1)
        self.theta = np.array([10.1, 11.1])
        self.examples_total: int = self.examples.shape[0]

    def hypotheses(self, x: np.ndarray[float]) -> float:
        return np.dot(self.theta, x)

    def loss_func(self) -> float:
        sum = 0

        for i in range(self.examples_total):
            x_i = self.examples[i, :][: self.feature_num + 1]
            y_i = self.examples[i, :][self.dim]
            sum += pow((self.hypotheses(x_i) - y_i), 2)

        return 0.5 * sum

    def partial_derivative_theta(self, j: int, example: np.ndarray[float]) -> float:
        end = example.size - 1

        x = example[: self.feature_num + 1]
        y = example[end]
        x_j = x[j]

        return (self.hypotheses(x) - y) * x_j

    def draw_loss_func(self):
        begin = -100
        end = 100
        total = 101

        X, Y = np.meshgrid(
            np.linspace(begin, end, total), np.linspace(begin, end, total)
        )

        Z = np.empty(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                self.theta = np.array([X[i, j], Y[i, j]])
                Z[i, j] = self.loss_func()

        fig = plt.figure()

        ax = fig.add_subplot(111, projection="3d")
        # ax.scatter(X, Y, Z, s=10)
        # ax.plot_surface(X, Y, Z)

        self.theta = np.array([10.0, 10.0])
        res = self.run()
        ax.plot(res[:, 0], res[:, 1], res[:, 2])
        ax.scatter(res[:, 0], res[:, 1], res[:, 2], s=10)
        print(res)

        ax.set_xlabel(r"$\theta_0$")
        ax.set_ylabel(r"$\theta_1$")
        ax.set_zlabel("Loss Function")

        # plt.contour(X, Y, Z)
        # plt.colorbar()

        # plt.savefig("loss_func.png", dpi=600, format="png")

        plt.show()

    def run(self) -> np.ndarray:
        res = np.empty((self.iter_total, 3))

        for k in range(self.iter_total):
            for j in range(self.theta.size):
                sum = 0
                for i in range(self.examples_total):
                    x_i = self.examples[i, :]
                    sum += self.partial_derivative_theta(j, x_i)
                self.theta[j] -= self.alpha * sum

                if j == 0:
                    res[k, 0] = self.theta[j]

                if j == 1:
                    res[k, 1] = self.theta[j]

            res[k, 2] = self.loss_func()

        return res


if __name__ == "__main__":
    machine = LinearRegressionMachine(
        data_file_path=DATA_FILE_PATH, alpha=ALPHA, feature_num=1, iter_total=ITER_TOTAL
    )

    # point_size = 2
    # x = machine.examples[:, 6]
    # y = machine.examples[:, machine.dim + 1]

    # plt.scatter(x, y, point_size)
    # plt.xlabel("RM")
    # plt.ylabel("MDEV")
    # plt.grid(True)
    # plt.savefig("test.png", dpi=600, format="png")

    # machine.run()

    machine.draw_loss_func()
