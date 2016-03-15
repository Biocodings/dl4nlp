import unittest
import numpy as np
from dl4nlp.gradient_descent import gradient_descent
from dl4nlp.sgd import bind_cost_gradient, batch_sampler
from dl4nlp.gradient_check import gradient_check


class TestSGD(unittest.TestCase):
    def test_supervised_gradient_descent(self):
        def linear_regression_cost_gradient(parameters, input, output):
            prediction = np.dot(parameters, input)
            cost = (prediction - output) ** 2
            gradient = 2.0 * (prediction - output) * input
            return cost, gradient

        inputs = np.random.normal(0.0, size=(10, 2))
        outputs = np.random.normal(0.0, size=10)
        initial_parameters = np.random.uniform(-1.0, 1.0, size=2)

        # Create cost and gradient function for supervised SGD and check its gradient
        cost_gradient = bind_cost_gradient(linear_regression_cost_gradient,
                                           inputs, outputs, sampler=batch_sampler)
        result = gradient_check(cost_gradient, initial_parameters)
        self.assertEqual([], result)

        # Run gradient descent on the function and see if it minimizes cost function
        actual, cost_history = gradient_descent(cost_gradient, initial_parameters, 10)

        # Compute exact solution of linear regression by closed form
        expected = np.linalg.solve(np.dot(inputs.T, inputs), np.dot(inputs.T, outputs))

        for e, a in zip(expected, actual):
            self.assertAlmostEqual(e, a, places=0)

    def test_supervised_gradient_descent_multi(self):
        def linear_regression_cost_gradient_multi(parameters, input, output):
            prediction = np.dot(parameters.T, input)
            cost = sum((prediction - output) ** 2)
            gradient = 2.0 * np.dot(input.reshape(-1, 1), (prediction - output).reshape(1, -1))
            return cost, gradient

        inputs = np.random.normal(0.0, size=(10, 2))
        outputs = np.random.normal(0.0, size=(10, 3))
        initial_parameters = np.random.uniform(-1.0, 1.0, size=(2, 3))

        # Create cost and gradient function for supervised SGD and check its gradient
        cost_gradient = bind_cost_gradient(linear_regression_cost_gradient_multi,
                                           inputs, outputs, sampler=batch_sampler)
        result = gradient_check(cost_gradient, initial_parameters)
        self.assertEqual([], result)

        # Run gradient descent on the function and see if it minimizes cost function
        actual, cost_history = gradient_descent(cost_gradient, initial_parameters, 10)

        # Compute exact solution of linear regression by closed form
        expected = np.linalg.solve(np.dot(inputs.T, inputs), np.dot(inputs.T, outputs))

        for e1, a1 in zip(expected, actual):
            for e2, a2 in zip(e1, a1):
                self.assertAlmostEqual(e2, a2)

if __name__ == '__main__':
    unittest.main()
