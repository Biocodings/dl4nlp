#!/usr/bin/env python
import argparse
import pickle
from dl4nlp.preprocessing import tokenize, lower, prepend_caret
from dl4nlp.nplm import NPLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType())
    parser.add_argument('output_file', type=argparse.FileType('wb'))
    parser.add_argument('vocabulary_size', type=int)
    parser.add_argument('context_size', type=int)
    parser.add_argument('feature_size', type=int)
    parser.add_argument('hidden_size', type=int)
    parser.add_argument('iterations', type=int)
    args = parser.parse_args()

    sentences = list(prepend_caret(lower(tokenize(args.input_file))))
    nplm = NPLM(args.vocabulary_size, args.feature_size, args.context_size, args.hidden_size)
    nplm.train(sentences, args.iterations)
    pickle.dump(nplm, args.output_file)

if __name__ == '__main__':
    main()
