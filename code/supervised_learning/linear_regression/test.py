import unittest
import main
import numpy as np


class TestCase(unittest.TestCase):
    def test_hypotheses(self):
        theta = np.array([1, 2])
        x = np.array([1])

        result = main.hypotheses(theta, x)

        self.assertEqual(result, 5)


if __name__ == "__main__":
    unittest.main()
