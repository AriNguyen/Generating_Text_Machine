# Text Processing Pipeline
We're using text preprocessing pipeline to accelerate the process of training model
[Pipeline for Python 3](https://github.com/AryaNguyen/Generating_Text_Machine/tree/master/pipelining)<br/>
[Pipeline for Python 2](https://github.com/AryaNguyen/Generating_Text_Machine/tree/master/pipelining_python2)

# Generating Text using Bidirectional LSTM & Word2Vec
Part 1: Train word predicting model 
save/words_vocab.pkl: hold [words, vocabs, vocab_to_int, int_to_vocab]
trained_model/my_model_generate_sentences.h5: model that predicts the best next word giving sequence of words

Part 2: Train sentence predicting model 
trained_model/doc2vec.w2v: gensim.models.Doc2Vec models
trained_model/my_model_sequence_lstm.80.hdf5: model that predicts the next best sentence vector giving sequence of sentences


