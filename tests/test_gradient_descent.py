import unittest
from dl4nlp.gradient_descent import *


class TestGradientDescent(unittest.TestCase):
    def test_gradient_descent(self):
        expected = 0.0

        def quadratic_cost_gradient(x):
            cost = (x - expected) ** 2
            gradient = 2.0 * (x - expected)
            return cost, gradient

        initial_parameters = np.random.normal()

        # Test gradient descent with constant learning rate
        final_parameters, cost_history = \
            gradient_descent(quadratic_cost_gradient, initial_parameters, 100, get_constant(0.5))
        self.assertAlmostEqual(expected, final_parameters, places=1)

        # Test gradient descent with AdaGrad learning rate
        final_parameters, cost_history = \
            gradient_descent(quadratic_cost_gradient, initial_parameters, 100, get_adagrad(0.5))
        self.assertAlmostEqual(expected, final_parameters, places=1)

if __name__ == '__main__':
    unittest.main()
