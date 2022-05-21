# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A vocabulary builder that generates vocab.txt to be used for training."""

import time
from typing import Iterable, Sequence

from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tftext

import vocabulary


def get_tokenized_sequences(
        dataset: tf.data.Dataset,
        tokenizer: tftext.Tokenizer = tftext.WhitespaceTokenizer(),
        input_key: str = 'sentence') -> Iterable[Sequence[bytes]]:
  """Returns tokenized sequences for vocabulary building."""
  dataset = dataset.map(
      lambda example: tokenizer.tokenize(example[input_key]),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  yield from tfds.as_numpy(dataset)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  start_time = time.time()

  # Loads the dataset to build the vocabulary from. We use the train split.
  dataset = tfds.load('glue/sst2', split='train')

  # Tokenizes the sequences in the dataset and keeps only those.
  tokenized_sequences = get_tokenized_sequences(dataset)

  # Builds the vocabulary from the tokenized sequences.
  # A token needs to appear at least 3 times to be in the vocabulary. You can
  # play with this. It is there to make sure we don't overfit on rare words.
  vocab = vocabulary.Vocabulary(
      tokenized_sequences=tokenized_sequences, min_freq=3)
  vocab.save('vocab.txt')

  logging.info('Total time elapsed: %f s', time.time() - start_time)
