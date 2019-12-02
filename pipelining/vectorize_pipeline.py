"""===============================================================================
  *
  * Author: Arya Nguyen
  * Component: pipelining
  * Description: convert to matrices
  *
=================================================================================="""

from .pipeline_step import PipelineStep
from keras.utils.np_utils import to_categorical


class VectorizeStep(PipelineStep):
    def __init__(self, sequence_length=None, **kwargs):
        super().__init__(**kwargs)
        self._sequence_length = sequence_length

    def execute(self, *args, sequence_length=None, **kwargs):
        super().__init__(*args, **kwargs)

        if sequence_length is not None:
            self._sequence_length = sequence_length

        if self._sequence_length is None:
            raise KeyError("KeyError: Missing arguemnt \"sequence_length\"")

        # Prepare the dataset of input to output pais encoded as integers
        dataX = []
        dataY = []
        for i in range(0, self.data.get_data_size() - self._sequence_length, 1):
            seq_in = self.data.get_word_list()[i: i + self._sequence_length]
            seq_out = self.data.get_word_list()[i + self._sequence_length]
            dataX.append([self.data.get_vocab_to_int()[word] for word in seq_in])
            dataY.append(self.data.get_vocab_to_int()[seq_out])

        self.data.set_dataX(dataX)

        X = to_categorical(dataX)
        self.data.set_sequences(X)

        # one hot encode the output variable
        y = to_categorical(dataY)
        self.data.set_next_words(y)
        return self