"""===============================================================================
  *
  * Author: Arya Nguyen
  * Component: pipelining
  * Description: Break sentence/full_text into words/tokens
  *
=================================================================================="""

from .pipeline_step import PipelineStep

import string
import spacy
nlp = spacy.load('en')


def create_word_list(doc):
    wl = []
    for word in doc:
        if word.text not in (r"\n", r"\n\n", r'\u2009', r'\xa0'):
            wl.append(word.text.lower())
    return wl


class TokenizeStep(PipelineStep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data = self.data.get_data().lower()

        for i in string.punctuation:
            data = ''.join(data.replace(i, ' {} '.format(i)))

        data = ' '.join(data.split())
        data = ''.join(data.splitlines())

        doc = nlp(data)
        self.data.set_word_list(create_word_list(doc))

        return self



