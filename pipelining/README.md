A PipelineStep can be used as a separated method or can be combined with other pipeline steps.
- WordTokenizer
- SentenceTokenizer
- Vectorizer

Use Pipeline:
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


Class PipelineData holds information about the data and gets updated after every pipeline step.
- WordTokenizer: After this step, PipelineData stores information about
    + word_list: list of all words in the text
    + vocabs: list of unique words in the text
    + vocabs_to_int: map of unique_word to integer value
    + int_to_vocabs: map of integer value to vocabulary
    + data_size: total number of

- SentenceTokenizer:
    + sentence_list: list of segmented sentence
    +


- Vectorizer:
    + sequence_size: size of the independent variable
    + X: vector of independent variable
    + y: vector of dependent variable

-