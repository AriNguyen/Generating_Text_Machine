# coding=utf-8
"""===============================================================================
  *
  * Author: Arya Nguyen
  * Component:Train LSTM + word2vec model
  * Description: Generating text using biLSTM and word2vec models
  *
=================================================================================="""

from __future__ import print_function
from six.moves import cPickle
from keras.models import load_model

from pipelining import TokenizeStep

import numpy as np
import os
import gensim
import scipy
import spacy
nlp = spacy.load('en')

global d2v_model
global vocab_size
global word_predictor_model
global sentence_predictor_model
global vocabs_to_int
global words
global vocab
global int_to_vocab


def load_models(vocab_model, doc2vec_model, word_predictor_model, sentence_predictor_model):
    """ Given models path, load models

    :param vocab_model:
    :param doc2vec_model:
    :param word_predictor_model:
    :param sentence_predictor_model:

    :return: list of 4 loaded models
    """

    # load the doc2vec model
    print("loading doc2Vec model...")
    d2v_model = gensim.models.doc2vec.Doc2Vec.load(doc2vec_model)
    print("model loaded!\n")

    # load vocabulary
    print("loading vocabulary...")
    with open(vocab_model, 'rb') as f:
        words, vocab, vocabs_to_int, int_to_vocab = cPickle.load(f)

    print("vocabulary loaded!\n")

    # load the keras word prediction model
    print("loading word prediction model...")
    word_predictor_model = load_model(word_predictor_model)
    print("model loaded!\n")

    # Load sentence selection model
    print("loading sentence selection model...")
    sentence_predictor_model = load_model(sentence_predictor_model)
    print("model loaded!\n")

    return words, vocab, vocabs_to_int, int_to_vocab, d2v_model, word_predictor_model, sentence_predictor_model


def sample(preds, temperature=1.0):
    """Sample an index from a probability array

    :param preds: prediction (ndarray)
    :param temperature (float): hyperparameter to tune predictions
        if == 1: prediction is not tuned
        if > 1: range of probabilities is shorten, more words are chosen
        if < 1: less words are chosen
    :return: index of the prediction (int)
    """
    # handling non-array-type preds: convert to array with a dtype of float64
    preds = np.asarray(preds).astype('float64')

    # tune preds: softmax function with temperature
    tuned_preds = np.log(preds)
    tuned_preds = tuned_preds / temperature
    exp_preds = np.exp(tuned_preds)
    tuned_preds = exp_preds / np.sum(exp_preds)

    # calculate probability: draw randomly 1 array of shape tuned_preds wit
    # probas = [[0, 0, 0,..., 1, ..., 0, 0]]
    probas = np.random.multinomial(1, tuned_preds, 1)

    # Returns the indices of the maximum values along an axis
    return np.argmax(probas)


def create_seed(seed_sentences, nb_words_in_seq=30, verbose=False):
    """ Given <seed_sentences>, return list of the last <nb_words_in_seq> items

    :param seed_sentences: (str)
    :param nb_words_in_seq: (int)
    :param verbose: (bool)

    :return: (list) of words

    """
    # Create list of tokens from seed_sentences
    tokenizer = TokenizeStep()
    tokenizer.execute(data=seed_sentences)
    word_list = tokenizer.data.get_word_list()

    # get last <nb_words_in_seq> items of input
    sentence = word_list[0 - nb_words_in_seq:]

    if verbose:
        print('Generating text with the following seed: "' + ' '.join(sentence) + '"')

    return sentence


def generate_phrase(sentence, max_words=50, nb_words_in_seq=30, temperature=1, verbose=False):
    """ Create next phrase of given sentence

    :param sentence: previous sentence (list) - last sentence of the paragraph
        list of words
    :param max_words: max word of generated sentence (int)
        default: 50
    :param nb_words_in_seq: len of <sentence>
    :param temperature: hyperparameter to tune predictions
    :param verbose:
    :return: str, list
        generated sentence, last <nb_words_in_seq> words of text and new generated sentence
    """
    generated = ""
    words_number = max_words - 1
    punctuation = [".", "?", "!", ":", "…"]
    seq_length = nb_words_in_seq
    # sentence = []
    is_punct = False

    # generate the text
    for i in range(words_number):
        # create the vector of <sentence>
        x = np.zeros((1, seq_length, vocab_size))
        for t, word in enumerate(sentence):
            # print(t, word, vocabs_to_int[word.lower()])
            x[0, nb_words_in_seq - len(sentence) + t, vocabs_to_int[word.lower()]] = 1.
        # print(x.shape)

        # calculate next word
        preds = word_predictor_model.predict(x, verbose=0)[0]

        # tune prediction and get the index of the next word
        next_index = sample(preds, temperature)
        next_word = int_to_vocab[next_index]

        if verbose:
            predv = np.array(preds)
            # arr = np.array([1, 3, 2, 4, 5])
            wi = predv.argsort()[-3:][::-1]
            print("potential next words: ", int_to_vocab[wi[0]], int_to_vocab[wi[1]], int_to_vocab[wi[2]])

        # add the next word to the text
        if not is_punct:
            if next_word in punctuation:
                is_punct = True
            generated += " " + next_word
            # shift the sentence by one, and and the next word at its end
            sentence = sentence[1:] + [next_word]

    return generated, sentence


def define_phrases_candidates(sentence,
                              max_words=50,
                              nb_words_in_seq=20,
                              temperature=1,
                              nb_candidates_sents=10,
                              verbose=False):
    """ Given a <sentence>, generate <nb_candidates_sents> potential sentences
    (best sentence after previous sentence)

    :param sentence: list
        previous sentence
    :param max_words: int
        max words of a potential sentence
    :param nb_words_in_seq: int
        number of word in sequence
    :param temperature: float
        hyperparameter to tune predictions
    :param nb_candidates_sents: int)
        number of potential sentences to generate
    :param verbose:

    :return: list([str, list], [], [])
        list of potential sentences (best sentence after previous sentence) to
    """
    phrase_candidate = []
    for i in range(nb_candidates_sents):
        generated_sentence, new_sentence = generate_phrase(sentence=sentence,
                                                           max_words=max_words,
                                                           nb_words_in_seq=nb_words_in_seq,
                                                           temperature=temperature,
                                                           verbose=False)
        phrase_candidate.append([generated_sentence, new_sentence])

    if verbose:
        for phrase in phrase_candidate:
            print("   ", phrase[0])
    return phrase_candidate


def create_sentences(doc):
    """

    :param doc:

    :return:
    """
    ponctuation = [".", "?", "!", ":", "…"]
    sentences = []
    sent = []
    for word in doc:
        if word.text not in ponctuation:
            if word.text not in ("\n", "\n\n", '\u2009', '\xa0'):
                sent.append(word.text.lower())
        else:
            sent.append(word.text.lower())
            if len(sent) > 1:
                sentences.append(sent)
            sent = []
    return sentences


def generate_training_vector(sentences_list, verbose=False):
    """ Generate training vector from list of sentence

    :param sentences_list: list of str
        list of sentences to infer vector
    :param verbose:

    :return:
    """
    if verbose:
        print("generate vectors for each sentence...")
    V = []  # list of inferred vectors of all sentence in <sentences_list>

    for sentence in sentences_list:
        # Create list of words from <sentence>
        tokenizer = TokenizeStep()
        tokenizer.execute(data=sentence)
        word_list = tokenizer.data.get_word_list()

        # infer the vector of the sentence, from the doc2vec model
        v = d2v_model.infer_vector(word_list, alpha=0.001, min_alpha=0.001, steps=10000)

        # create the vector array for the model
        V.append(v)
    V_val = np.array(V)

    # expand dimension to fit the entry of the model : that's the training vector
    V_val = np.expand_dims(V_val, axis=0)

    if verbose:
        print("Vectors generated!")

    return V_val


def select_next_phrase(model, V_val, candidate_list, verbose=False):
    """

    :param model:
    :param V_val:
    :param candidate_list:
    :param verbose:

    :return:
    """
    sims_list = []

    # calculate prediction
    preds = model.predict(V_val, verbose=0)[0]

    # calculate vector for each candidate
    for candidate in candidate_list:
        # calculate vector
        # print("calculate vector for : ", candidate[1])
        V = np.array(d2v_model.infer_vector(candidate[1]))
        # calculate csonie similarity
        sim = scipy.spatial.distance.cosine(V, preds)
        # populate list of similarities
        sims_list.append(sim)

    # select index of the biggest similarity
    m = max(sims_list)
    index_max = sims_list.index(m)

    if verbose:
        print("selected phrase :")
        print("     ", candidate_list[index_max][0])
    return candidate_list[index_max]


def generate_paragraph(phrase_seed,
                       sentences_seed,
                       max_words=50,
                       nb_words_in_seq=20,
                       temperature=1,
                       nb_phrases=30,
                       nb_candidates_sents=10,
                       verbose=True):
    """ Generate sentences based on given seed sentences

    :param phrase_seed: seed sentence to predict the next word (list)
        list of words
    :param sentences_seed: seed sequence of sentences (list)
        list of sentences
    :param max_words: max word of new generated sentence (int)
        default: 50
    :param nb_words_in_seq: number of words to keep as seed for next word prediction
    :param temperature: float
        hyperparameter to tune predictions
    :param nb_phrases: int
        number of sentence to generate
    :param nb_candidates_sents: number of candidates of sentences to generate for each new sentence
    :param verbose: whether print process (boolean)

    :return: list of generated senetences
    """
    sentences_list = sentences_seed
    sentence = phrase_seed
    text = []

    for p in range(nb_phrases):
        if verbose:
            print("")
            print("#############")
        print("phrase ", p + 1, "/", nb_phrases)
        if verbose:
            print("#############")
            print('Sentence to generate phrase : ')
            print("     ", sentence)
            print("")
            print('List of sentences to constrain next phrase : ')
            print("     ", sentences_list)
            print("")

        # generate seed training vectors
        V_val = generate_training_vector(sentences_list=sentences_list, verbose=verbose)

        # generate phrase candidates
        if verbose:
            print("generate phrases candidates...")
        phrases_candidates = define_phrases_candidates(sentence=sentence,
                                                       max_words=max_words,
                                                       nb_words_in_seq=nb_words_in_seq,
                                                       temperature=temperature,
                                                       nb_candidates_sents=nb_candidates_sents,
                                                       verbose=verbose)
        # Select next best sentence (among <phrases_candidates>)
        if verbose:
            print("select next phrase...")
        next_phrase = select_next_phrase(model=sentence_predictor_model,
                                         V_val=V_val,
                                         candidate_list=phrases_candidates,
                                         verbose=verbose)
        print("Next phrase: ", next_phrase[0])

        if verbose:
            print("")
            print("Shift phrases in sentences list...")
        for i in range(len(sentences_list) - 1):
            sentences_list[i] = sentences_list[i + 1]

        sentences_list[len(sentences_list) - 1] = next_phrase[0]

        if verbose:
            print("done.")
            print("new list of sentences :")
            print("     ", sentences_list)
        sentence = next_phrase[1]

        text.append(next_phrase[0])

    return text


# set directories and file path
save_dir = 'save'  # directory where models are stored
vocabs_file_path = os.path.join(os.getcwd(), "save/words_vocab.pkl")
doc2vec_file_path = os.path.join(os.getcwd(), 'trained_model/doc2vec.w2v')
word_prediction_model_path = os.path.join(os.getcwd(), 'trained_model/my_model_gen_sentences.92-0.04.hdf5')
sentence_selection_model_path = os.path.join(os.getcwd(), 'trained_model/my_model_sequence_lstm.80.hdf5')

# Load models
words, vocab, vocabs_to_int, int_to_vocab, d2v_model, word_predictor_model, sentence_predictor_model = load_models(
    vocabs_file_path,
    doc2vec_file_path,
    word_prediction_model_path,
    sentence_selection_model_path
)
vocab_size = len(vocab)

# Create 15 make-sense sentence to ...
s1 = r"Poor me!"
s2 = r"As Alice said this she looked down at her hands, and was surprised to see that she had put on one of the Rabbit's little white kid gloves while she was talking."
s3 = r"'How CAN I have done that?' she thought."
s4 = r"As she said these words her foot slipped, and in another moment, splash!"
s5 = r"she was up to her chin in salt water."
s6 = r"Just then she heard something splashing about in the pool a little way off, and she swam nearer to make out what it was."
s7 = r"at first she thought it must be a walrus or hippopotamus, but then she remembered how small she was now, and she soon made out that it was only a mouse that had slipped in like herself."
s8 = r"'Would it be of any use, now,' thought Alice, 'to speak to this mouse?"
s9 = r"Everything is so out-of-the-way down here, that I should think very likely it can talk: at any rate, there's no harm in trying.' So she began:"
s10 = r"O Mouse, do you know the way out of this pool? I am very tiredof swimming about here, O Mouse!'"
s11 = r"(Alice thought this must be the right way of speaking to a mouse: she had never done such a thing before, but she remembered having seen in her brother's Latin Grammar, 'A mouse--of a mouse--to a mouse--a mouse--O mouse!')"
s12 = r"The Mouse looked at her rather inquisitively, and seemed to her to wink with one of its little eyes, but it said nothing."
s13 = r"'Perhaps it doesn't understand English,' thought Alice; 'I daresay it's a French mouse, come over with William the Conqueror.'"
s14 = r"So she began again: 'Ou est ma chatte?' which was the first sentence in her French lesson-book."
s15 = r"The Mouse gave a sudden leap out of the water, and seemed to quiver all over with fright."

sentences_list = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15]

# combine list of sentences to a phrase
combined_sentences_list = ''
for i in sentences_list:
    combined_sentences_list += i
    combined_sentences_list += " "

#
sentences_seed = create_seed(
    seed_sentences=combined_sentences_list,
    nb_words_in_seq=30
)

text = generate_paragraph(phrase_seed=sentences_seed,
                          sentences_seed=sentences_list,
                          max_words=50,
                          nb_words_in_seq=30,
                          temperature=2,
                          nb_phrases=20,
                          nb_candidates_sents=15,
                          verbose=True)

a