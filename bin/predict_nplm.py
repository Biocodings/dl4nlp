#!/usr/bin/env python
import sys
import argparse
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('rb'))
    args = parser.parse_args()

    nplm = pickle.load(args.input_file)
    print('Type context words:')
    for line in sys.stdin:
        line = line.rstrip('\n')
        context = ['^', '^'] + [word.lower() for word in line.split(' ')]
        word = nplm.predict(context)
        print(word)

if __name__ == '__main__':
    main()
