import unittest
from dl4nlp.preprocessing import tokenize, lower, prepend_caret, build_dictionary, to_indices


class TestPreprocessing(unittest.TestCase):
    def test_tokenize(self):
        lines = ['I am beautiful.', ' !?']
        actual = list(tokenize(lines))
        expected = [['I', 'am', 'beautiful']]
        self.assertSequenceEqual(expected, actual)

    def test_lower(self):
        sentences = [['I', 'am', 'beautiful']]
        actual = list(lower(sentences))
        expected = [['i', 'am', 'beautiful']]
        self.assertSequenceEqual(expected, actual)

    def test_prepend_caret(self):
        sentences = [['i', 'am', 'beautiful']]
        actual = list(prepend_caret(sentences))
        expected = [['^', 'i', 'am', 'beautiful']]
        self.assertSequenceEqual(expected, actual)

    def test_build_dictionary(self):
        sentences = [['a', 'man', 'a', 'woman'], ['a', 'man']]
        actual = build_dictionary(sentences, 2)
        expected = {'a': 1, 'man': 2}
        self.assertDictEqual(expected, actual)

    def test_to_indices(self):
        sentences = [['a', 'man', 'a', 'woman'], ['a', 'man']]
        dictionary = build_dictionary(sentences, 2)
        actual = list(to_indices(sentences, dictionary))
        expected = [[1, 2, 1, 0], [1, 2]]
        self.assertSequenceEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()
