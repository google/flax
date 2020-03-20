# SST-2 Sentiment Analysis example in Flax

## Sentiment analysis in a nutshell

The task is to classify an input example (e.g., a sentence) into a sentiment (here: positive or negative).
We will use the SST-2 dataset as used in the GLUE benchmark.

An example from the data set:
```python
{
  'sentence': 'this cross-cultural soap opera is painfully formulaic and stilted .',
  'label': 0
}
```

In this example we will build and train a **text classifier** on this data set. 

Each sentence is mapped to a sequence of integers using a vocabulary.
The vocabulary is just a dictionary that maps each word that occurs in SST-2 to a unique ID.  That looks like this for the above sentence:

```python
{
  'sentence':  [2, 4, 28, 10199, 3817, 4527, 13, 7380, 11530, 7, 12366, 5, 3],
  'label': 0
}
```
Note that a `2` was added at the beginning to mark the start, and a `3` to mark the end of the sequence.

## Model
Our model consists of word embeddings, an LSTM encoder, and an MLP classifier.

```
                               0.9    Prediction
                                ^
                                |
                              [...]   MLP
                                ^
                                |
[...]---->[...]---->[...]---->[...]   LSTM
  ^         ^         ^         ^
  |         |         |         |
This      movie      was      great
```

We will use pre-trained [GloVe word embeddings](https://nlp.stanford.edu/projects/glove/) to map each word ID to a 300-dimensional word embedding vector.
The LSTM reads in the sentence word by word (that is, embedding by embedding), updating its hidden state at each time step.
The MLP takes the final hidden state from the LSTM as input and outputs a scalar prediction. After taking a sigmoid, we treat `output > 0.5` as a positive classification, and `output <= 0.5` as negative.


## Historic note on SST-2
This dataset consisted of annotated *trees* originally, with sentiment labels (very negative to very positive) at every node of the tree.  There are only 6920 training trees with non-neutral root labels, and to make more use of the  dataset the yields of sub-trees were also included in the training set. (Exactly how that was done seems to be unknown at this point.) So that means that the training data contains a lot of overlapping phrases!

## Instructions

1. Download and unzip the pre-trained word embeddings:

```sh
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
```

2. Train the model as follows. The SST-2 data is downloaded from TFDS and cached
   on the first run.

```sh
python3 train.py --glove-path glove.840B.300d.txt
```


