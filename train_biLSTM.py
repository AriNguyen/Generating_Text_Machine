"""===============================================================================
  *
  * Author: Arya Nguyen
  * Component:
  * Description: Generating text using biLSTM and word2vec
  *
=================================================================================="""

from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.metrics import categorical_accuracy
from matplotlib import pyplot

from pipelining import Pipeline, TokenizeStep, VectorizeStep

import os

# Define variables
data_dir = 'text/wonderland.txt'
save_dir = 'save'
sequences_step = 1
sequence_length = 30
input_file = os.path.join(os.getcwd(), data_dir)

rnn_size = 256  # size of RNN
learning_rate = 0.001  # learning rate
batch_size = 128  # batch size
num_epochs = 100  # number of epochs

filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-biLSTM.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Define Pipeline Steps
tokenize_data = TokenizeStep()

vectorize_data = VectorizeStep(sequence_length=sequence_length)

steps = [
    ('Tokenize text', tokenize_data),
    ('Vectorize text', vectorize_data)
]

pipeline = Pipeline(data=input_file, steps=steps)
pipeline.execute()

X = pipeline.data.get_sequences()
y = pipeline.data.get_next_words()


def bidirectional_lstm_model(seq_length, n_vocab):
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_size, activation="relu"), input_shape=(seq_length, n_vocab)))
    model.add(Dropout(rate=0.2))
    model.add(Dense(n_vocab, activation='softmax'))

    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])

    return model


md = bidirectional_lstm_model(sequence_length, pipeline.data.get_vocabs_size())

# Fit the model
history = md.fit(X, y,
                 batch_size=batch_size,
                 epochs=num_epochs,
                 callbacks=callbacks_list)

