#!/usr/bin/env python
import argparse
import operator
import numpy as np
from dl4nlp.word2vec import skip_gram_cost_gradient, create_context
from dl4nlp.sgd import bind_cost_gradient, get_stochastic_sampler
from dl4nlp.gradient_descent import gradient_descent
from dl4nlp.preprocessing import tokenize, lower, build_dictionary, to_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType())
    parser.add_argument('output_file', type=argparse.FileType('w'))
    parser.add_argument('vector_size', type=int)
    parser.add_argument('context_size', type=int)
    parser.add_argument('vocabulary_size', type=int)
    args = parser.parse_args()

    sentences = list(lower(tokenize(args.input_file)))
    dictionary = build_dictionary(sentences, args.vocabulary_size)
    indices = to_indices(sentences, dictionary)
    inputs, outputs = create_context(indices, args.context_size)

    cost_gradient = bind_cost_gradient(skip_gram_cost_gradient, inputs, outputs, sampler=get_stochastic_sampler(100))
    initial_parameters = np.random.normal(size=(2, len(dictionary) + 1, args.vector_size))
    parameters, cost_history = gradient_descent(cost_gradient, initial_parameters, 10000)
    input_vectors, output_vectors = parameters
    word_vectors = input_vectors + output_vectors
    sorted_pairs = sorted(dictionary.items(), key=operator.itemgetter(1))
    words = [word for word, index in sorted_pairs]

    for word in words:
        vector = word_vectors[dictionary[word]]
        vector_string = ' '.join(str(element) for element in vector)
        print(word, vector_string, file=args.output_file)

if __name__ == '__main__':
    main()
