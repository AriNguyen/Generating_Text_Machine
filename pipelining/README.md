# Pipeline 
A **PipelineStep** can be used as a separated method or can be combined with other pipeline steps.
- WordTokenizer
- SentenceTokenizer
- Vectorizer

**Use Pipeline:**
```
import Pipeline

text = ""
pipeline = Pipeline(
    data=text,
    steps=[
        WordTokenizer(),
        SentenceTokenizer(),
        Vectorizer()
    ]
)
pipeline.execute()
```

Class **PipelineData** holds information about the data and gets updated after every pipeline step.<br/>
**WordTokenizer**: After this step, PipelineData stores information about<br/>
    + word_list: list of all words in the text<br/>
    + vocabs: list of unique words in the text<br/>
    + vocabs_to_int: map of unique_word to integer value<br/>
    + int_to_vocabs: map of integer value to vocabulary<br/>
    + data_size: total number of

**SentenceTokenizer**:<br/>
    + sentence_list: list of segmented sentence<br/>

**Vectorizer**:<br/>
    + sequence_size: size of the independent variable<br/>
    + X: vector of independent variable<br/>
    + y: vector of dependent variable<br/>
