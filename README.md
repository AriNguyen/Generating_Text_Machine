# Text Preprocessing Pipeline
I'm using text preprocessing pipeline to accelerate the process of training model. Instructions on how to use can be found [here](https://github.com/AryaNguyen/Generating_Text_Machine/blob/develop/pipelining/README.md) 
- [Pipeline for Python 3](https://github.com/AryaNguyen/Generating_Text_Machine/tree/master/pipelining)<br/>
- [Pipeline for Python 2](https://github.com/AryaNguyen/Generating_Text_Machine/tree/master/pipelining_python2)<br/>


# Generating Text using Bidirectional Long Short-Term Memmory (LSTM) & Word2Vec</br>
In this project, I utilize Bidirectional LSTM & Word2vec to learn the writers' choice of words and sentences in their literary works.<br/> 
I use "Alice's Adventures in Wonderland" by Lewis Carroll as dataset in experiment 1 and "THE SONNETS" by William Shakespeare" in experiment 2. The full text of these 2 literature can be found in **text** folder. You can download the files from [Project Gutenberg](https://www.gutenberg.org/) library where this literature is no longer protected by copyright.<br/>

**Step 1: Train word predicting model**<br/>
- save/words_vocab.pkl: hold [words, vocabs, vocab_to_int, int_to_vocab]<br/>
- trained_model/my_model_generate_sentences.h5: model that predicts the best next word giving sequence of words<br/>

**Step 2: Train sentence predicting model**<br/>
- trained_model/doc2vec.w2v: gensim.models.Doc2Vec models<br/>
- trained_model/my_model_sequence_lstm.80.hdf5: model that predicts the next best sentence vector giving sequence of sentences<br/>

**Step3: Generate text**

