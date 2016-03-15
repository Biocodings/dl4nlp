# DL4NLP: Deep Learning for Natural Language Processing
## Introduction
DL4NLP is a Python package inspired by a course at Stanford University by Richard Socher.

http://cs224d.stanford.edu/

## Requirements
The recommended environment is Anaconda with Python 3.4 and PyCharm for development.

- https://www.continuum.io/
- https://www.jetbrains.com/pycharm/

For matrix computation, it uses only numpy i.e. you don't need Theano or other framework.

## word2vec
DL4NLP implements word2vec model (skip-gram) proposed by Thomas Mikolov at 2013.

https://code.google.com/archive/p/word2vec/

In order to train word vectors, use this command:

    ./bin/word2vec.py input.txt output.txt <vector_size> <context_size>


## Unit testing
Run the following command on the top of project directory.

    python -m unittest discover tests
