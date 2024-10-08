import numpy as np
import math

import sys

sys.path.append("../../..")

import math_utils.linalg as linalg

MAX = sys.float_info.max


class MultiClassificationMachine:
    def __init__(
        self,
        data: np.ndarray,
        class_num: int,
        feature_num: int,
        alpha: float,
        iter_total: int,
    ) -> None:
        # Examples is the input of X, where x_0 = 1
        self.__examples: np.ndarray = np.hstack((np.ones((data.shape[0], 1)), data))
        self.__alpha = alpha  # Learning rate
        self.__dim = self.__examples.shape[1] - 1  # Num of columns in X
        self.__class_num = class_num
        self.__feature_num = feature_num if feature_num <= self.__dim else self.__dim
        self.__iter_total = iter_total
        # The parameters of hypotheses, i.e. h(x) = \theta^T * x
        self.__theta = np.random.rand(self.__class_num, self.__feature_num + 1) / 1000
        self.__examples_total: int = self.__examples.shape[0]

    def p(self, y, x) -> float:
        p_list = np.dot(self.__theta, x)

        for i in range(p_list.size):
            try:
                p_list[i] = math.exp(p_list[i])
            except OverflowError:
                p_list[i] = MAX

        sum = 0
        for p in p_list:
            sum += p

        return p_list[int(y)] / sum

    def partial_derivative_theta(self, i: int, example: np.ndarray[float]) -> float:
        end = example.size - 1

        x = example[: self.__feature_num + 1]
        y = example[end]

        phi_i = self.p(i, x)

        return (phi_i - (1 if y == i else 0)) * x

    def likehood(self) -> float:
        res = 0.0
        log = math.log
        p = self.p

        for i in range(self.__examples_total):
            x_i = self.__examples[i, :][: self.__feature_num + 1]
            y_i = self.__examples[i, :][self.__dim]

            phi = p(y_i, x_i)
            if phi == 0.0:
                res -= MAX
            else:
                res += log(phi)

        return -res

    def batch_gradient_descent(self) -> list[float]:
        res = [self.likehood()]

        for k in range(self.__iter_total):
            for j in range(self.__theta.shape[0]):
                pd = np.zeros(self.__feature_num + 1)
                for i in range(self.__examples_total):
                    example = self.__examples[i, :]
                    pd += self.partial_derivative_theta(j, example)

                self.__theta[j] -= self.__alpha * pd

            res.append(self.likehood())

        return res

    def stochastic_gradient_descent(self) -> list[float]:
        """
        Find the value of theta minimizes likehood
        """

        res = [self.likehood()]

        for k in range(self.__iter_total):
            row = np.empty((1, self.__feature_num + 2))

            for i in range(self.__examples_total):
                example = self.__examples[i, :]

                for j in range(self.__theta.shape[0]):
                    pd = self.partial_derivative_theta(j, example)
                    self.__theta[j] -= self.__alpha * pd

                res.append(self.likehood())

        return res

    def fisher_scoring(self) -> list[float]:
        """
        Use Newton Method to minimize likehood
        """

        res = [self.likehood()]

        for k in range(self.__iter_total):
            for i in range(self.__theta.shape[0]):
                # Hessian matrix
                H = np.zeros((self.__feature_num + 1, self.__feature_num + 1))
                pd = np.zeros(self.__feature_num + 1)

                for j in range(self.__examples_total):
                    example = self.__examples[j, :]
                    x = example[: self.__feature_num + 1]

                    e = [
                        math.exp(np.dot(self.__theta[d], x))
                        for d in range(self.__theta.shape[0])
                    ]
                    e_sum = sum(e)
                    H_i = (e[i] * (e_sum - e[i]) / e_sum**2) * x.reshape(-1, 1) * x

                    H = np.add(H, H_i)
                    pd += self.partial_derivative_theta(i, example)

                self.__theta[i] -= np.dot(linalg.inv(H), pd)
                res.append(self.likehood())

        return res
