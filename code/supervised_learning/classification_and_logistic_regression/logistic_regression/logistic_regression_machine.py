import numpy as np
import math


class LogisticRegressionMachine:
    def __init__(
        self,
        data: np.ndarray,
        alpha: float,
        feature_num: int,
        iter_total: int,
    ) -> None:
        # Examples is the input of X, where x_0 = 1
        self.__examples: np.ndarray = np.hstack((np.ones((data.shape[0], 1)), data))
        self.__alpha = alpha  # Learning rate
        self.__dim = self.__examples.shape[1] - 1  # Num of columns in X
        self.__feature_num = feature_num if feature_num <= self.__dim else self.__dim
        self.__iter_total = iter_total
        # The parameters of hypotheses, i.e. h(x) = \theta^T * x
        # self.__theta = np.random.uniform(0, 10, self.__feature_num + 1)
        self.__theta = np.random.rand(self.__feature_num + 1)
        self.__examples_total: int = self.__examples.shape[0]

    def set_theta(self, theta: np.ndarray) -> None:
        if theta.shape != self.__theta.shape:
            raise Exception(
                "The shape of input ({}) is not equal to theta ({})".format(
                    theta.shape, self.__theta.shape
                )
            )

        self.__theta = theta

    def h(self, x: np.ndarray[float]) -> float:
        """
        hypotheses
        """
        z = np.dot(self.__theta, x)
        return 1 / (1 + pow(math.e, -z))

    def p(self, y, x) -> float:
        """
        probability
        """
        return pow(self.h(x), y) * pow(1 - self.h(x), 1 - y)

    def likehood(self) -> float:
        res = 0.0
        log = math.log
        h = self.h

        for i in range(self.__examples_total):
            x_i = self.__examples[i, :][: self.__feature_num + 1]
            y_i = self.__examples[i, :][self.__dim]
            h_x_i = h(x_i)
            res += y_i * log(h_x_i) + (1 - y_i) * log(1 - h_x_i)

        return -res

    def partial_derivative_theta(self, j: int, example: np.ndarray[float]) -> float:
        end = example.size - 1

        x = example[: self.__feature_num + 1]
        y = example[end]
        x_j = x[j]

        return (self.h(x) - y) * x_j

    def batch_gradient_descent(self) -> np.ndarray:
        res = np.empty((1, self.__feature_num + 2))
        for i in range(self.__theta.size):
            res[0, i] = self.__theta[i]
        res[0, self.__theta.size] = self.likehood()

        for k in range(self.__iter_total):
            row = np.empty((1, self.__feature_num + 2))

            for j in range(self.__theta.size):
                pd = 0.0
                for i in range(self.__examples_total):
                    x_i = self.__examples[i, :]
                    pd += self.partial_derivative_theta(j, x_i)

                self.__theta[j] -= self.__alpha * pd
                row[0, j] = self.__theta[j]

            row[0, self.__theta.size] = self.likehood()
            res = np.vstack((res, row))

        return res
