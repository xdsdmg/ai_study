import numpy as np
import math


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
        # self.__theta = np.random.uniform(0, 10, self.__feature_num + 1)
        self.__theta = np.random.rand(self.__class_num, self.__feature_num + 1)
        self.__examples_total: int = self.__examples.shape[0]

    def p(self, y, x) -> float:
        p_list = np.dot(self.__theta, x)

        for i in range(p_list.size):
            p_list[i] = pow(math.e, p_list[i])

        sum = 0
        for p in p_list:
            sum += p

        return p_list[int(y)] / sum

    def partial_derivative_theta(self, i: int, example: np.ndarray[float]) -> float:
        end = example.size - 1

        x = example[: self.__feature_num + 1]
        y = example[end]

        phi_i = self.p(y, x)

        return (phi_i - (1 if y == i else 0)) * x

    def likehood(self) -> float:
        res = 0.0
        log = math.log
        p = self.p

        for i in range(self.__examples_total):
            x_i = self.__examples[i, :][: self.__feature_num + 1]
            y_i = self.__examples[i, :][self.__dim]
            res += log(p(y_i, x_i))

        return -res

    def batch_gradient_descent(self) -> np.ndarray:
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
