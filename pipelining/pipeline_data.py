"""===============================================================================
  *
  * Author: Arya Nguyen
  * Component: pipelining
  * Description: Abstract base class that defines the pipeline step
  *
=================================================================================="""

import os
import codecs
from six.moves import cPickle


def save_wordlist_vocabs(vocab_file_path, words, vocabs, vocab_to_int, int_to_vocab):
    with open(vocab_file_path, 'wb') as f:
        cPickle.dump((words, vocabs, vocab_to_int, int_to_vocab), f)


class PData:
    def __init__(self, data=None):
        self._data = data
        self._word_list = None  # list of all words from data
        self._vocabs = None  # list of unique words
        self._vocab_to_int = None  # dict of {vocab: int}
        self._int_to_vocab = None  # dict of {integer: vocab}
        self._vocabs_size = None  # num of vocabs (unique word) in text
        self._data_size = None  # num of words in text
        self._sequences = None
        self._next_words = None
        self._sequences_size = None
        self._dataX = None

    def set_data(self, data):
        self._data = data

    def set_word_list(self, word_list, vocab_file_path):
        self._word_list = word_list
        self._data_size = len(word_list)

        # list of sorted unique words, punctuation, number
        self._vocabs = sorted(list(set(self._word_list)))
        self._vocabs_size = len(self._vocabs)
        self._vocab_to_int = dict((word, integer) for integer, word in enumerate(self._vocabs))
        self._int_to_vocab = dict((integer, word) for integer, word in enumerate(self._vocabs))

        if vocab_file_path is not None:
            save_wordlist_vocabs(
                vocab_file_path=vocab_file_path,
                words=self._word_list,
                vocabs=self._vocabs,
                vocab_to_int=self._vocab_to_int,
                int_to_vocab=self._int_to_vocab
            )

    def set_sequences(self, sequences):
        self._sequences = sequences
        self._sequences_size = len(sequences)

    def set_dataX(self, dataX):
        self._dataX = dataX

    def get_data(self):
        return self._data

    def get_word_list(self):
        return self._word_list

    def get_vocabs(self):
        return self._vocabs

    def get_vocab_to_int(self):
        return self._vocab_to_int

    def get_int_to_vocab(self):
        return self._int_to_vocab

    def get_vocabs_size(self):
        return self._vocabs_size

    def get_data_size(self):
        return self._data_size

    def get_sequences_size(self):
        return self._sequences_size

    def get_sequences_matrix(self):
        return self._sequences

    def set_next_words(self, n):
        self._next_words = n

    def get_next_words_matrix(self):
        return self._next_words

    def get_dataX(self):
        return self._dataX



