import unittest
import numpy as np
from scipy.special import expit
from dl4nlp.gradient_check import gradient_check
from dl4nlp.utilities import sigmoid_gradient


class TestGradientCheck(unittest.TestCase):
    def test_gradient_check(self):
        def quad(x):
            return np.sum(x ** 2), x * 2

        self.assertEqual([], gradient_check(quad, np.array(123.456)))      # scalar test
        self.assertEqual([], gradient_check(quad, np.random.randn(3,)))    # 1-D test
        self.assertEqual([], gradient_check(quad, np.random.randn(4,5)))   # 2-D test

    def test_gradient_check_sigmoid(self):
        def sigmoid_check(x):
            return expit(x), sigmoid_gradient(expit(x))

        x = np.array(0.0)
        result = gradient_check(sigmoid_check, x)
        self.assertEqual([], result)

if __name__ == '__main__':
    unittest.main()
