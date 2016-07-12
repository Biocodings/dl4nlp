import unittest
import numpy as np
from math import exp
from scipy.special import expit
from dl4nlp.utilities import sigmoid_gradient, softmax


class TestUtilities(unittest.TestCase):
    def assertDistribution(self, distribution):
        self.assertTrue(all(distribution >= 0.0))
        self.assertTrue(all(distribution <= 1.0))
        self.assertEqual(1.0, np.sum(distribution))

    def assertNumpyEqual(self, expect, actual):
        self.assertEqual(expect.shape, actual.shape)
        if expect.shape == ():  # This is scalar!
            self.assertAlmostEqual(expect, actual)
        else:   # This is array
            for e, a in zip(expect, actual):
                self.assertNumpyEqual(e, a)

    def test_softmax(self):
        # softmax should receive numpy array and return normalized vector
        expect = np.array([exp(1) / (exp(1) + exp(2)), exp(2) / (exp(1) + exp(2))])
        actual = softmax(np.array([1, 2]))
        self.assertDistribution(actual)
        self.assertNumpyEqual(expect, actual)

        # softmax should be invariant to constant offsets in the input
        # softmax should be able to handle very large or small values
        actual = softmax(np.array([1001, 1002]))
        self.assertNumpyEqual(expect, actual)
        actual = softmax(np.array([-1002, -1001]))
        self.assertNumpyEqual(expect, actual)

        # softmax should receive matrix and return matrix of same size
        expect = np.array([[exp(1) / (exp(1) + exp(2)), exp(2) / (exp(1) + exp(2))],
                           [exp(1) / (exp(1) + exp(2)), exp(2) / (exp(1) + exp(2))]])
        actual = softmax(np.array([[1, 2], [3, 4]]))
        self.assertNumpyEqual(expect, actual)

    def test_sigmoid(self):
        x = np.array([[1, 2], [-1, -2]])
        f = expit(x)
        g = sigmoid_gradient(f)
        expected = np.array([[0.73105858,  0.88079708],
                    [0.26894142,  0.11920292]])
        self.assertNumpyEqual(expected, f)

        expected = np.array([[0.19661193,  0.10499359],
                    [0.19661193,  0.10499359]])
        self.assertNumpyEqual(expected, g)

if __name__ == '__main__':
    unittest.main()
