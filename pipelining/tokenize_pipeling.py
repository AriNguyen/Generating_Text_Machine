"""===============================================================================
  *
  * Author: Arya Nguyen
  * Component: pipelining
  * Description: Break sentence/full_text into words/tokens
  *
=================================================================================="""

from .pipeline_step import PipelineStep

import nltk
import string
import warnings
import spacy
nlp = spacy.load('en')


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '{}:{}:\n    {}: {}\n'.format(filename, lineno, category.__name__, message)


class TokenizeStep(PipelineStep):
    def __init__(self, vocab_file_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_file_path = vocab_file_path

    def execute(self, vocab_file_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if vocab_file_path is not None:
            self.vocab_file_path = vocab_file_path

        # convert all to lower case
        data = self.data.get_data().lower()

        # Warn user about lowering all words in text
        warnings.formatwarning = warning_on_one_line
        warnings.warn('Text input is transformed to lowercase')

        #
        for i in ['s', 'll', 've', 're', 'm', 't', 'd']:
            data = ''.join(data.replace(r"'{} ".format(i), r" '{} ".format(i)))

        # tokenize text string
        word_list = nltk.word_tokenize(data)

        new_word_list = []
        for index, i in enumerate(word_list):
            if i in [r"'s", r"'ll", r"'ve", r"'re", r"'m", r"'t", r"'d"]:
                new_word_list.append(i)
                continue
            elif i[0] == r"'":
                new_word_list.append(i[0])
                new_word_list.append(i[1:])
                continue
            else:
                new_word_list.append(i)

        self.data.set_word_list(word_list=new_word_list, vocab_file_path=self.vocab_file_path)

        return self

