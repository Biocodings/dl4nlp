import unittest
from dl4nlp.word2vec import *
from dl4nlp.gradient_descent import *
from dl4nlp.sgd import *
from dl4nlp.gradient_check import *


class TestWord2vec(unittest.TestCase):
    def test_softmax_cost_gradient(self):
        vocabulary_size = 5
        vector_size = 2
        input = np.random.randint(0, vocabulary_size)
        output = np.random.randint(0, vocabulary_size)

        def softmax_cost_gradient_wrapper(parameters):
            cost, gradient = softmax_cost_gradient(parameters, input, output)
            return cost, gradient

        initial_parameters = np.random.normal(size=(2, vocabulary_size, vector_size))
        result = gradient_check(softmax_cost_gradient_wrapper, initial_parameters)
        self.assertEqual([], result)

    def test_skip_gram(self):
        vocabulary_size = 5
        vector_size = 2
        context_size = 3
        data_size = 4
        inputs = np.random.randint(0, vocabulary_size, size=data_size)
        outputs = np.random.randint(0, vocabulary_size, size=(data_size, context_size))
        initial_parameters = np.random.normal(size=(2, vocabulary_size, vector_size))

        # Create cost and gradient function for supervised SGD and check its gradient
        cost_gradient = bind_cost_gradient(skip_gram_cost_gradient, inputs, outputs, sampler=batch_sampler)
        result = gradient_check(cost_gradient, initial_parameters)
        self.assertEqual([], result)
        final_parameters, cost_history = gradient_descent(cost_gradient, initial_parameters, 10)

if __name__ == '__main__':
    unittest.main()
