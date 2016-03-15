import unittest
import numpy as np
from dl4nlp.nplm import NPLM


class TestNPLM(unittest.TestCase):
    def test_nplm(self):
        np.random.seed(0)

        # Test settings
        vocabulary_size = 5
        context_size = 1
        hidden_size = 1
        feature_size = 2

        # Gradient check
        data_size = 4
        inputs = np.random.randint(0, vocabulary_size, size=(data_size, context_size))
        outputs = np.random.randint(0, vocabulary_size, size=(data_size, 1))
        nplm = NPLM(vocabulary_size, feature_size, context_size, hidden_size)
        result = nplm.gradient_check(inputs, outputs)
        self.assertEqual([], result)

        # Train NPLM
        sentences = [['^', 'i', 'am'], ['^', 'you', 'are']]
        nplm = NPLM(vocabulary_size, feature_size, context_size, hidden_size)
        nplm.train(sentences, 100)

        # Check if next word is predicted from context
        for sentence in sentences:
            context = sentence[:-1]
            word = sentence[-1]
            prediction = nplm.predict(context)
            self.assertEqual(word, prediction)

if __name__ == '__main__':
    unittest.main()
