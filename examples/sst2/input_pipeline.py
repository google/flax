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

"""SST-2 input pipeline."""

from typing import Any, Dict, Optional

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text

import vocabulary


AUTOTUNE = tf.data.experimental.AUTOTUNE
Example = Dict[str, tf.Tensor]


def get_bucket_boundaries(bucket_size: int, max_size: int) -> np.ndarray:
  """Bucket boundaries with `bucket_size` items per bucket, up to `max_size`.

  Example:
  ```
  get_bucket_boundaries(8, 24)
  [9, 17, 25]
  ```
  E.g., the first boundary covers items with sizes 0-8, the next boundary covers
  items with sizes 9-16, and the last bucket covers sizes 17-24. Each bucket
  covers 8 different sizes (e.g., sentence lengths).

  Args:
   bucket_size: The number of different items per bucket.
   max_size: The maximum size to be expected for a bucket.

  Returns:
    A list of (exclusive) bucket boundaries.
  """
  return np.arange(bucket_size, max_size + bucket_size, bucket_size) + 1


def get_num_examples(dataset: tf.data.Dataset) -> int:
  """Returns the number of examples in the dataset by iterating over it once."""
  return dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()


def get_bucketed_batches(
    dataset: tf.data.Dataset,
    batch_size: int,
    bucket_size: int,
    max_length: int,
    padded_shapes: Any,
    example_size_fn: Any,
    shuffle: bool = False,
    shuffle_seed: Optional[int] = None,
    drop_remainder: bool = False,
) -> tf.data.Dataset:
  """Returns padded batches of shuffled examples bucketed by length.

  This shuffles the examples randomly each epoch. The random order is
  deterministic and controlled by the seed.

  Batches are padded because sentences have different lengths.
  Sentences that are shorter in a batch will get 0s added at the end, until
  all sentences in the batch have the same length.

  For performance, examples of similar lengths are bucketed together. However,
  the contents of the buckets and their order is random each epoch, and
  controlled by the seed.

  Args:
    dataset: A TF Dataset with SST examples to be shuffled and batched.
    batch_size: The size of each batch. The remainder is dropped.
    bucket_size: How many different lengths go in each bucket.
    max_length: The maximum length to provide a bucket for.
    padded_shapes: A nested structure representing the shape to which the
      respective component of each input element should be padded prior to
      batching. See `tf.data.Dataset.padded_batch` for examples.
    example_size_fn: A TF function that returns the size of an example to
      determine in which bucket it goes. E.g., the sentence length.
    shuffle: Shuffle the dataset each epoch using seed.
    shuffle_seed: The seed that determines the shuffling order, with a
      different order each epoch.
    drop_remainder: Drop the last batch if it is not of size batch_size.

  Returns:
    A TF Dataset containing padded bucketed batches.
  """
  if shuffle:
    assert shuffle_seed is not None, 'When shuffling you must provide a seed.'

  # For bucket_size 8 and max length 24, we get bucket boundaries [9, 17, 25].
  bucket_boundaries = get_bucket_boundaries(bucket_size, max_length)
  logging.info('Batching bucket boundaries: %r', bucket_boundaries)

  # One batch size for each bucket plus one additional one (per requirement).
  bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
  bucket_fn = tf.data.experimental.bucket_by_sequence_length(
      example_size_fn,
      bucket_boundaries,
      bucket_batch_sizes,
      padded_shapes=padded_shapes,
      pad_to_bucket_boundary=True,
      drop_remainder=drop_remainder)

  if shuffle:
    # For shuffling we need to know how many training examples we have.
    num_examples = get_num_examples(dataset)
    num_batches = num_examples // batch_size
    return dataset.shuffle(
        num_examples, seed=shuffle_seed,
        reshuffle_each_iteration=True).apply(bucket_fn).shuffle(
            num_batches,
            seed=shuffle_seed,
            reshuffle_each_iteration=True).prefetch(
                tf.data.experimental.AUTOTUNE)
  return dataset.apply(bucket_fn).prefetch(tf.data.experimental.AUTOTUNE)


def vocab_to_hashtable(vocab: vocabulary.Vocabulary,
                       unk_idx: int) -> tf.lookup.StaticHashTable:
  """Returns a TF lookup table (token -> ID) from a vocabulary."""
  return tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          list(vocab.keys()), list(vocab.values())), default_value=unk_idx)


def vocab_to_inverse_hashtable(vocab: vocabulary.Vocabulary,
                               unk_token: bytes) -> tf.lookup.StaticHashTable:
  """Returns an inverse TF lookup table (ID -> token) from a vocabulary."""
  return tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          list(vocab.values()),
          list(vocab.keys()),
          key_dtype=tf.int64,
          value_dtype=tf.string),
      default_value=unk_token)


def _is_text_field(feature_name_and_type):
  """Identifies a text field when given a feature (name, type) pair."""
  _, feature_type = feature_name_and_type
  return isinstance(feature_type, tfds.features.Text)


def _is_class_label(feature_name_and_type):
  """Identifies a class label field when given a feature (name, type) pair."""
  _, feature_type = feature_name_and_type
  return isinstance(feature_type, tfds.features.ClassLabel)


class TextDataset:
  """A text dataset with one sequence as input and a label."""

  def __init__(self,
               tfds_name: str = 'glue/sst2',
               vocab_path: str = 'vocab.txt',
               tokenizer: text.Tokenizer = text.WhitespaceTokenizer(),
               split='train'):
    """Initializes the SST2 data source."""
    self.dataset, self.info = tfds.load(tfds_name, split=split, with_info=True)

    # Look up the feature name of the text and label in the dataset.
    # We assume there is one text input and one label.
    text_fields = filter(_is_text_field, self.info.features.items())
    label_fields = filter(_is_class_label, self.info.features.items())
    self.text_feature_name, _ = next(text_fields)
    self.label_feature_name, _ = next(label_fields)

    # Load the vocabulary.
    self.vocab = vocabulary.Vocabulary(vocab_path=vocab_path)

    # Convert the sentences to sequences of token IDs and compute length.
    self.tokenizer = tokenizer
    self.tf_vocab = vocab_to_hashtable(self.vocab, unk_idx=self.vocab.unk_idx)
    self.examples = self.dataset.map(
        self.prepare_example, num_parallel_calls=AUTOTUNE).cache()

  @property
  def padded_shapes(self):
    """The padded shapes used by batching functions."""
    # None means variable length; pads to the longest sequence in the batch.
    return {'idx': [], 'token_ids': [None], 'label': [], 'length': []}

  def example_length_fn(self, example: Example) -> tf.Tensor:
    """Returns the length of the example for the purpose of the bucketing."""
    return tf.size(example['token_ids'])

  def add_bos_eos(self, sequence: tf.Tensor) -> tf.Tensor:
    """Prepends BOS ID and appends EOS ID to a sequence of token IDs."""
    return tf.concat(
        [[self.vocab.bos_idx], sequence, [self.vocab.eos_idx]], 0)

  def prepare_example(self, example: Example) -> Example:
    """Prepares an example by converting text to token IDs."""
    tokens = self.tokenizer.tokenize(example[self.text_feature_name])
    label = example[self.label_feature_name]
    del example[self.text_feature_name]
    del example[self.label_feature_name]
    example['token_ids'] = self.add_bos_eos(self.tf_vocab.lookup(tokens))
    example['length'] = tf.size(example['token_ids'])
    example['label'] = label
    return example

  def get_batches(self,
                  batch_size: int,
                  drop_remainder: bool = False,
                  shuffle: bool = False,
                  shuffle_seed: Optional[int] = None,
                  fixed_pad_length: Optional[int] = None,
                  dataset: Optional[tf.data.Dataset] = None):
    """Returns an iterator with padded batches for the provided dataset."""
    if dataset is None:
      dataset = self.examples
    if shuffle:
      buffer_size = get_num_examples(dataset)
      dataset = dataset.shuffle(
          buffer_size, seed=shuffle_seed, reshuffle_each_iteration=True)
    padded_shapes = {k: v for k, v in self.padded_shapes.items()}
    if fixed_pad_length is not None:
      padded_shapes['token_ids'] = fixed_pad_length
    return dataset.padded_batch(
        batch_size, padded_shapes=padded_shapes, drop_remainder=drop_remainder)

  def get_bucketed_batches(self,
                           batch_size: int,
                           bucket_size: int,
                           max_input_length: int,
                           drop_remainder: bool = False,
                           shuffle: bool = False,
                           shuffle_seed: Optional[int] = None,
                           dataset: Optional[tf.data.Dataset] = None):
    """Returns an iterator with bucketed batches for the provided dataset."""
    if dataset is None:
      dataset = self.examples
    return get_bucketed_batches(
        dataset,
        batch_size,
        bucket_size,
        max_input_length,
        self.padded_shapes,
        self.example_length_fn,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        drop_remainder=drop_remainder)
