import unittest
import numpy as np
from scipy.special import expit
from dl4nlp.logistic_regression import logistic_regression_cost_gradient, multinomial_logistic_regression_cost_gradient
from dl4nlp.gradient_descent import gradient_descent
from dl4nlp.sgd import bind_cost_gradient, batch_sampler, get_stochastic_sampler
from dl4nlp.gradient_check import gradient_check
from dl4nlp.utilities import softmax


class TestLogisticRegression(unittest.TestCase):
    def test_logistic_regression(self):
        input = np.random.uniform(-10.0, 10.0, size=10)
        output = np.random.randint(0, 2)

        def logistic_regression_wrapper(parameters):
            return logistic_regression_cost_gradient(parameters, input, output)

        initial_parameters = np.random.normal(scale=1e-5, size=10)
        result = gradient_check(logistic_regression_wrapper, initial_parameters)
        self.assertEqual([], result)

        # Train logistic regression and see if it predicts correct label
        final_parameters, cost_history = gradient_descent(logistic_regression_wrapper, initial_parameters, 100)
        prediction = expit(np.dot(input, final_parameters)) > 0.5
        self.assertEqual(output, prediction)

    def test_multinomial_logistic_regression(self):
        input_size = 10
        output_size = 5
        input = np.random.normal(size=(input_size,))
        output = np.random.randint(0, output_size)

        def multinomial_logistic_regression_wrapper(parameters):
            return multinomial_logistic_regression_cost_gradient(parameters, input, output)

        initial_parameters = np.random.normal(size=(input_size, output_size))
        result = gradient_check(multinomial_logistic_regression_wrapper, initial_parameters)
        self.assertEqual([], result)

        # Train multinomial logistic regression and see if it predicts correct label
        final_parameters, cost_history = gradient_descent(
            multinomial_logistic_regression_wrapper, initial_parameters, 100)
        prediction = softmax(np.dot(final_parameters.T, input)) > 0.5
        for i in range(len(prediction)):
            if output == i:
                self.assertEqual(1, prediction[i])
            else:
                self.assertEqual(0, prediction[i])

    def assertLogisticRegression(self, sampler):
        data_size = 3
        input_size = 5
        inputs = np.random.uniform(-10.0, 10.0, size=(data_size, input_size))
        outputs = np.random.randint(0, 2, size=data_size)
        initial_parameters = np.random.normal(scale=1e-5, size=input_size)

        # Create cost and gradient function for gradient descent and check its gradient
        cost_gradient = bind_cost_gradient(logistic_regression_cost_gradient,
                                           inputs, outputs, sampler=sampler)
        result = gradient_check(cost_gradient, initial_parameters)
        self.assertEqual([], result)

        # Train logistic regression and see if it predicts correct labels
        final_parameters, cost_history = gradient_descent(cost_gradient, initial_parameters, 100)
        predictions = expit(np.dot(inputs, final_parameters)) > 0.5

        # Binary classification of 3 data points with 5 dimension is always linearly separable
        for output, prediction in zip(outputs, predictions):
            self.assertEqual(output, prediction)

    def test_batch_logistic_regression(self):
        self.assertLogisticRegression(batch_sampler)

    def test_stochastic_logistic_regression(self):
        self.assertLogisticRegression(get_stochastic_sampler(2))

    def assertMultinomialLogisticRegression(self, sampler):
        data_size = 3
        input_size = 5
        output_size = 4
        inputs = np.random.uniform(-10.0, 10.0, size=(data_size, input_size))
        outputs = np.random.randint(0, output_size, size=data_size)
        initial_parameters = np.random.normal(size=(input_size, output_size))

        # Create cost and gradient function for gradient descent and check its gradient
        cost_gradient = bind_cost_gradient(multinomial_logistic_regression_cost_gradient,
                                           inputs, outputs, sampler=sampler)
        result = gradient_check(cost_gradient, initial_parameters)
        self.assertEqual([], result)

        # Train multinomial logistic regression and see if it predicts correct labels
        final_parameters, cost_history = gradient_descent(cost_gradient, initial_parameters, 100)
        predictions = np.argmax(softmax(np.dot(final_parameters.T, inputs.T)), axis=0)

        for output, prediction in zip(outputs, predictions):
            self.assertEqual(output, prediction)

    def test_batch_multinomial_logistic_regression(self):
        self.assertMultinomialLogisticRegression(batch_sampler)

    def test_stochastic_multinomial_logistic_regression(self):
        self.assertMultinomialLogisticRegression(get_stochastic_sampler(2))


if __name__ == '__main__':
    unittest.main()
