"""===============================================================================
  *
  * Author: Arya Nguyen
  * Component:
  * Description: Generating text using biLSTM and word2vec
  *
=================================================================================="""
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.metrics import categorical_accuracy

from pipelining import Pipeline, TokenizeStep, VectorizeStep

import os
import spacy
nlp = spacy.load('en')

# Define variables
data_dir = 'text/wonderland.txt'
save_dir = 'save'
vocab_file = os.path.join(os.getcwd(), save_dir, 'words_vocab')
sequences_step = 1
input_file = os.path.join(os.getcwd(), data_dir)


def bidirectional_lstm_model(seq_length, vocab_size):
    print('Build LSTM model.')
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_size, activation="relu"), input_shape=(seq_length, vocab_size)))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    print("model built!")
    return model


# Define Pipeline Steps
tokenize_data = TokenizeStep(vocab_file_path=vocab_file)
vectorize_data = VectorizeStep(sequence_length=30)

steps = [
    ('Tokenize text', tokenize_data),
    ('Vectorize step', vectorize_data)
]

pipeline = Pipeline(data=input_file, steps=steps)
pipeline.execute()

data = pipeline.data

#
vocab_size = data.get_vocabs_size()
X = data.get_sequences_matrix()
y = data.get_next_words_matrix()

batch_size = 128  # batch size
num_epochs = 100  # number of epochs
rnn_size = 256  # size of RNN
seq_length = 30  # sequence length
learning_rate = 0.001  # learning rate

# Create model
md = bidirectional_lstm_model(seq_length, vocab_size)

callbacks = [
    ModelCheckpoint(filepath=save_dir + '/my_model_gen_sentences.{epoch:02d}-{val_loss:.2f}.hdf5',
                    monitor='loss',
                    verbose=1,
                    save_best_only=True,
                    mode='min')
]

# fit the model
history = md.fit(X, y,
                 batch_size=batch_size,
                 shuffle=True,
                 epochs=num_epochs,
                 callbacks=callbacks,
                 validation_split=0.1)

# save the model
md.save(save_dir + '/my_model_generate_sentences.h5')
