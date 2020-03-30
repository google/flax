# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SST-2 input pipeline."""

import collections
from typing import Dict, List, Set, Text, Tuple
import os

from absl import logging

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def build_vocab(datasets: Sequence[tf.data.Dataset],
                special_tokens: Sequence[Text] = (b'<pad>', b'<unk>', b'<s>', b'</s>'),
                min_freq: int = 0
) -> Dict[Text, int]:
  """Returns a vocabulary of tokens with optional minimum frequency."""
  # Count the tokens in the datasets.
  counter = Counter()
  for dataset in datasets:
    for example in tfds.as_numpy(dataset):
      counter.update(whitespace_tokenize(example['text']))

  # Add special tokens to the start of vocab.
  vocab = OrderedDict()
  for token in special_tokens:
    vocab[token] = len(vocab)

  # Add all other tokens to the vocab if their frequency is >= min_freq.
  for token in sorted(list(counter.keys())):
    if counter[token] >= min_freq:
      vocab[token] = len(vocab)

  logging.info('Number of unfiltered tokens: %d', len(counter))
  logging.info('Vocabulary size: %d', len(vocab))
  return vocab


def whitespace_tokenize(s: Text) -> Sequence[Text]:
  """Splits an input into tokens by whitespace."""
  return s.strip().split()


def get_shuffled_batches(dataset: tf.data.Dataset,
                         seed: int = 0,
                         batch_size: int = 64) -> tf.data.Dataset:
  """Returns a Dataset that consists of padded batches when iterated over.

  This shuffles the examples randomly each epoch. The random order is
  deterministic and controlled by the seed.

  Batches are padded because sentences have different lengths.
  Sentences that are shorter in a batch will get 0s added at the end, until
  all sentences in the batch have the same length.

  Args:
    dataset: A TF Dataset with examples to be shuffled and batched.
    seed: The seed that determines the shuffling order, with a different order
      each epoch.
    batch_size: The size of each batch. The remainder is dropped.

  Returns:
    A TF Dataset containing padded batches.
  """
  # For shuffling we need to know how many training examples we have.
  num_examples = dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()

  # `padded_shapes` says what kind of shapes to expect: [] means a scalar, [-1]
  # means a vector of variable length, and [1] means a vector of size 1.
  return dataset.shuffle(
      num_examples, seed=seed, reshuffle_each_iteration=True).padded_batch(
          batch_size,
          padded_shapes={
              'text': [-1],
              'label': [1],
              'length': []
          },
          drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)


def get_batches(dataset: tf.data.Dataset,
                batch_size: int = 64) -> tf.data.Dataset:
  """Returns a Dataset that consists of padded batches when iterated over."""
  return dataset.padded_batch(
      batch_size,
      padded_shapes={
          'text': [-1],
          'label': [1],
          'length': []
      },
      drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)


class SST2DataSource(object):
  """Provides SST-2 data as pre-processed batches, a vocab, and embeddings."""

  def __init__(self,
               batch_size: int = None,
               train_path: Text = None,
               valid_path: Text = None,
               test_path: Text = None,
               min_freq: int = 0,
               seed: int = 0,
               **kwargs):
    del kwargs

    # Load datasets.
    train_raw = open_as_tf_dataset(train_path)
    valid_raw = open_as_tf_dataset(valid_path)
    test_raw = open_as_tf_dataset(test_path)

    # Print an example.
    logging.info('Data sample: %s', next(tfds.as_numpy(train_raw.skip(4))))

    # Get a vocabulary and a corresponding GloVe word embedding matrix.
    vocab = build_vocab((train_raw,), min_freq=min_freq)

    unk_idx = vocab[b'<unk>']
    bos_idx = vocab[b'<s>']
    eos_idx = vocab[b'</s>']

    # Turn data examples into pre-processed examples by turning each sentence
    # into a sequence of token IDs. Also pre-prepend a beginning-of-sequence
    # token <s> and append an end-of-sequence token </s>.

    def tokenize(text: tf.Tensor):
      """Whitespace tokenize text."""
      return [whitespace_tokenize(text.numpy())]

    def tf_tokenize(text: tf.Tensor):
      return tf.py_function(tokenize, [text], Tout=tf.string)

    def encode(tokens: tf.Tensor):
      """Encodes a sequence of tokens (strings) into a sequence of token IDs."""
      return [[vocab[t] if t in vocab else unk_idx for t in tokens.numpy()]]

    def tf_encode(tokens: tf.Tensor):
      """Maps tokens to token IDs."""
      return tf.py_function(encode, [tokens], Tout=tf.int64)

    def tf_wrap_sequence(sequence: tf.Tensor):
      """Prepends BOS ID and appends EOS ID to a sequence of token IDs."""
      return tf.concat(([bos_idx], tf.concat((sequence, [eos_idx]), 0)), 0)

    def preprocess_example(example: Dict[Text, tf.Tensor]):
      example['text'] = tf_wrap_sequence(
          tf_encode(tf_tokenize(example['text'])))
      example['label'] = [example['label']]
      example['length'] = tf.shape(example['text'])[0]
      return example

    self.preprocess_fn = preprocess_example

    # Pre-process all datasets.
    self.train_dataset = train_raw.map(preprocess_example).cache()
    self.valid_dataset = valid_raw.map(preprocess_example).cache()
    self.test_dataset = test_raw.map(preprocess_example).cache()

    self.valid_raw = valid_raw
    self.test_raw = test_raw

    self.vocab = vocab
    self.vocab_size = len(vocab)

    self.bos_idx = bos_idx
    self.eos_idx = eos_idx
    self.unk_idx = unk_idx

