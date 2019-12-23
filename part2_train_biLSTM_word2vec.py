"""===============================================================================
  *
  * Author: Arya Nguyen
  * Component: Train LSTM + word2vec model
  * Description: predicts the next best sentence vector giving sequence of sentences
  *
=================================================================================="""
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

import gensim
from gensim.models.doc2vec import TaggedDocument

from six.moves import cPickle

import matplotlib.pyplot as plt

import numpy as np
import os
import time
import codecs
import nltk
import spacy
nlp = spacy.load('en')


def bidirectional_lstm_model(seq_length, vector_dim):
    print('Building LSTM model...')
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_size, activation="relu"), input_shape=(seq_length, vector_dim)))
    model.add(Dropout(0.2))
    model.add(Dense(vector_dim))

    optimizer = Adam(lr=learning_rate)
    callbacks = [EarlyStopping(patience=2, monitor='val_loss')]
    model.compile(loss='logcosh', optimizer=optimizer, metrics=['acc'])
    print('LSTM model built.')
    return model


def train_doc2vec_model(data, docLabels, size=300, sample=0.000001, dm=0, hs=1, window=10, min_count=0, workers=8,
                        alpha=0.024, min_alpha=0.024, epoch=15, save_file='./trained_model/doc2vec.w2v'):
    startime = time.time()

    print("{0} articles loaded for model".format(len(data)))

    it = LabeledLineSentence(data, docLabels)
    for i in it:
        print(i)

    model = gensim.models.Doc2Vec(
        size=size,
        sample=sample,
        dm=dm,
        window=window,
        min_count=min_count,
        workers=workers,
        alpha=alpha,
        min_alpha=min_alpha,
        hs=hs
    )  # use fixed learning rate

    model.build_vocab(it)
    for epoch in range(epoch):
        print("Training epoch {}".format(epoch + 1))
        model.train(it, total_examples=model.corpus_count, epochs=model.iter)
        # model.alpha -= 0.002 # decrease the learning rate
        # model.min_alpha = model.alpha # fix the learning rate, no decay

    # saving the created model
    model.save(os.path.join(os.getcwd(), save_file))
    print('model saved')


class LabeledLineSentence:
    def __init__(self, sentences_list, sentences_label, verbose=False):
        """
        :param sentences_list: list of list of words in sentence

        """
        self.sentences_list = sentences_list
        self.sentences_label = sentences_label

    def __iter__(self):
        for idx, sentence in enumerate(self.sentences_list):
            yield TaggedDocument(sentence, [self.sentences_label[idx]])


if __name__ == '__main__':
    # Define variables
    data_dir = 'text/wonderland.txt'
    save_dir = 'save'
    vocab_file = os.path.join(save_dir, 'words_vocab.pkl')
    sequences_step = 1
    input_file = os.path.join(os.getcwd(), data_dir)

    # read data
    with codecs.open(input_file, "r", encoding='utf-8-sig') as f:
        data = f.read()
    data = data.replace("\r\n", " ")

    # create list of sentences
    sentences = nltk.sent_tokenize(data)
    sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]

    # create labels
    sentences_label = []
    for i in range(np.array(sentences).shape[0]):
        sentences_label.append("ID" + str(i))

    train_doc2vec_model(
        sentences, sentences_label,
        size=500,
        sample=0.0,
        alpha=0.025,
        min_alpha=0.001,
        min_count=0,
        window=10,
        epoch=20,
        dm=0,
        hs=1,
        save_file='trained_model/doc2vec.w2v'
    )

    # load the model
    d2v_model = gensim.models.doc2vec.Doc2Vec.load('trained_model/doc2vec.w2v')

    sentences_vector = []
    t = 200
    for i in range(len(sentences)):
        if i % t == 0:
            print("sentence", i, ":", sentences[i])
            print("***")
        sent = sentences[i]
        sentences_vector.append(d2v_model.infer_vector(sent, alpha=0.001, min_alpha=0.001, steps=10000))

    """# save the sentences_vector
    sentences_vector_file = os.path.join(save_dir, "sentences_vector_500_a001_ma001_s10000.pkl")
    with open(os.path.join(sentences_vector_file), 'wb') as f:
        cPickle.dump((sentences_vector), f)"""

    nb_sequenced_sentences = 15
    vector_dim = 500

    X_train = np.zeros((len(sentences), nb_sequenced_sentences, vector_dim), dtype=np.float)
    y_train = np.zeros((len(sentences), vector_dim), dtype=np.float)

    #
    t = 200
    for i in range(len(sentences_label) - nb_sequenced_sentences - 1):
        if i % t == 0:
            print("new sequence: ", i)

        for k in range(nb_sequenced_sentences):
            sent = sentences_label[i + k]
            vect = sentences_vector[i + k]

            if i % t == 0:
                print("  ", k + 1, "th vector for this sequence. Sentence ", sent, "(vector dim = ", len(vect), ")")

            for j in range(len(vect)):
                X_train[i, k, j] = vect[j]

        senty = sentences_label[i + nb_sequenced_sentences]
        vecty = sentences_vector[i + nb_sequenced_sentences]
        if i % t == 0:
            print("  y vector for this sequence ", senty, ": (vector dim = ", len(vecty), ")")
        for j in range(len(vecty)):
            y_train[i, j] = vecty[j]

    print(X_train.shape, y_train.shape)

    rnn_size = 512  # size of RNN
    vector_dim = 500
    learning_rate = 0.0001  # learning rate

    sentence_predictor_model = bidirectional_lstm_model(nb_sequenced_sentences, vector_dim)

    batch_size = 128  # minibatch size

    callbacks = [
        ModelCheckpoint(filepath=save_dir + 'my_model_sequence_lstm.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5',
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        mode='min')
    ]

    history = sentence_predictor_model.fit(X_train, y_train,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           epochs=1000,
                                           callbacks=callbacks,
                                           validation_split=0.1)

    # save the model
    sentence_predictor_model.save(save_dir + "/" + 'my_model_sequence_lstm.final2.hdf5')

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')


