import numpy as np


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
        self.__dim = self.__examples.shape[1] - 1  # Num of columns in X
        self.__feature_num = feature_num if feature_num <= self.__dim else self.__dim
        self.__examples_total: int = self.__examples.shape[0]

        self.__mu_0 = 0.0
        self.__mu_1 = 0.0
        self.__sigma = 0.0
        self.__phi = 0.0
