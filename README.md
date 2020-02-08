# Text Processing Pipeline
We're using text preprocessing pipeline to accelerate the process of training model<br/>
Pipeline instructions can be found at [here](https://github.com/AryaNguyen/Generating_Text_Machine/blob/develop/pipelining/README.md) 
- [Pipeline for Python 3](https://github.com/AryaNguyen/Generating_Text_Machine/tree/master/pipelining)<br/>
- [Pipeline for Python 2](https://github.com/AryaNguyen/Generating_Text_Machine/tree/master/pipelining_python2)<br/>


# Generating Text using Bidirectional LSTM & Word2Vec
**Step 1: Train word predicting model**<br/>
- save/words_vocab.pkl: hold [words, vocabs, vocab_to_int, int_to_vocab]<br/>
- trained_model/my_model_generate_sentences.h5: model that predicts the best next word giving sequence of words<br/>

**Step 2: Train sentence predicting model**<br/>
- trained_model/doc2vec.w2v: gensim.models.Doc2Vec models<br/>
- trained_model/my_model_sequence_lstm.80.hdf5: model that predicts the next best sentence vector giving sequence of sentences<br/>


