import numpy as np


def det(matrix: np.ndarray) -> float:
    if matrix.shape[0] != matrix.shape[1]:
        raise Exception(
            f"The current matrix is not a square matrix (rows: {matrix.shape[0]}, columns: {matrix.shape[1]}), only a square matrix can calculate the determinant."
        )

    d = matrix.shape[0]

    if d == 1:
        return matrix[0][0]
    elif d == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        res = 0.0
        for col in range(d):
            sub_matrix = np.array(
                [[matrix[row][k] for k in range(d) if k != col] for row in range(1, d)]
            )
            sign = (-1) ** col
            res += sign * matrix[0][col] * det(sub_matrix)
        return res


def adj(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] != matrix.shape[1]:
        raise Exception(
            f"The current matrix is not a square matrix (rows: {matrix.shape[0]}, columns: {matrix.shape[1]}), only a square matrix can calculate the adjoint matrix."
        )

    d = matrix.shape[0]

    res = np.empty(matrix.shape)

    for i in range(d):
        for j in range(d):
            sub_matrix = np.array(
                [
                    [matrix[row][col] for col in range(d) if col != i]
                    for row in range(d)
                    if row != j
                ]
            )
            res[i][j] = (-1) ** (i + j) * det(sub_matrix)

    return res


def inv(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] != matrix.shape[1]:
        raise Exception(
            f"The current matrix is not a square matrix (rows: {matrix.shape[0]}, columns: {matrix.shape[1]}), only a square matrix can calculate the inverse matrix."
        )

    return (1 / det(matrix)) * adj(matrix)
