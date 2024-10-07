import linalg
import numpy as np


def test_det():
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"numpy: {np.linalg.det(m)}, mine: {linalg.det(m)}")


def test_inverse():
    m = np.array([[1, 2, 5], [4, 5, 6], [7, 8, 9]])
    print(f"numpy:\n{np.linalg.inv(m)}\n\nmine:\n{linalg.inv(m)}\n")
    print(f"adj:\n{linalg.adj(m)}\n")
    print(f"mul:\n{np.dot(m,linalg.inv(m))}")


if __name__ == "__main__":
    test_inverse()
