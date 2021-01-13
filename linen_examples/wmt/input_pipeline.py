# Copyright 2021 The Flax Authors.
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

"""Input pipeline for a WMT dataset."""

import os
from typing import Dict, Optional, List, Union

from absl import logging
import tokenizer
import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE


# -----------------------------------------------------------------------------
# Raw TFDS dataset.
# -----------------------------------------------------------------------------
def raw_wmt_datasets(dataset_name='wmt17_translate/de-en',
                     eval_dataset_name=None,
                     reverse_translation=False,
                     shard_idx=0,
                     shard_count=1):
  """Load raw WMT datasets and normalize feature keys.

  Args:
    dataset_name: str: TFDS WMT dataset name.
    eval_dataset_name: Optional[str]: separate dataset name for evaluation. e.g.
      for specifying the standard academic WMT14 test set.
    reverse_translation: bool: whether to reverse the translation direction.
      e.g. for 'de-en' this translates from english to german.
    shard_idx: int: for multihost training, index of this host.
    shard_count: int: for mulithost training, number of total hosts.

  Returns:
    training tf.dataset, evaluation tf.dataset, and training features_info
    source and target language features are mapped to 'inputs' and 'targets'
    keys.
  """
  builder = tfds.builder(dataset_name)
  shard_spec = (f'[{int(100 * shard_idx / shard_count)}%'
                f':{int(100 * (shard_idx + 1) / shard_count)}%]')
  logging.info('Training on TFDS dataset %s with split %s', dataset_name,
               'train' + shard_spec)
  train_data = builder.as_dataset(
      split='train' + shard_spec, shuffle_files=True)
  if eval_dataset_name is None:
    logging.info('Evaluating on TFDS dataset %s with split %s', dataset_name,
                 'validation' + shard_spec)
    eval_data = builder.as_dataset(
        split='validation' + shard_spec, shuffle_files=False)
  else:
    eval_dataset, *eval_split = eval_dataset_name.split(':')
    if not eval_split:
      eval_split = 'validation'
    else:
      eval_split = eval_split[0]
    logging.info('Evaluating on TFDS dataset %s with split %s', eval_dataset,
                 eval_split + shard_spec)
    eval_builder = tfds.builder(eval_dataset)
    eval_data = eval_builder.as_dataset(
        split=eval_split + shard_spec, shuffle_files=False)

  features_info = builder.info

  # standardize on 'inputs' and 'targets' features.
  input_lang = features_info.supervised_keys[0]
  target_lang = features_info.supervised_keys[1]
  if reverse_translation:
    input_lang, target_lang = target_lang, input_lang

  def to_features_dict(x):
    return {'inputs': x[input_lang], 'targets': x[target_lang]}

  train_data = train_data.map(to_features_dict, num_parallel_calls=AUTOTUNE)
  eval_data = eval_data.map(to_features_dict, num_parallel_calls=AUTOTUNE)

  return train_data, eval_data, features_info


def pack_dataset(dataset: tf.data.Dataset,
                 key2length: Union[int, Dict[str, int]],
                 keys: Optional[List[str]] = None) -> tf.data.Dataset:
  """Creates a 'packed' version of a dataset on-the-fly.

  Adapted from the mesh-tf implementation.

  This is meant to replace the irritation of having to create a separate
  "packed" version of a dataset to train efficiently on TPU.
  Each example in the output dataset represents several examples in the
  input dataset.
  For each key in the input dataset, two additional keys are created:
  <key>_segmentation: an int32 tensor identifying the parts
     representing the original example.
  <key>_position: an int32 tensor identifying the position within the original
     example.
  Example:
  Two input examples get combined to form an output example.
  The input examples are:
  {"inputs": [8, 7, 1, 0], "targets":[4, 1, 0]}
  {"inputs": [2, 3, 4, 1], "targets":[5, 6, 1]}
  The output example is:
  {
                 "inputs": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
    "inputs_segmentation": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
        "inputs_position": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                "targets": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
   "targets_segmentation": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
       "targets_position": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
  }
  0 represents padding in both the inputs and the outputs.
  Sequences in the incoming examples are truncated to length "length", and the
  sequences in the output examples all have fixed (padded) length "length".

  Args:
    dataset: a tf.data.Dataset
    key2length: an integer, or a dict from feature-key to integer
    keys: a list of strings (e.g. ["inputs", "targets"])

  Returns:
    a tf.data.Dataset
  """
  shapes = tf.nest.map_structure(lambda spec: spec.shape, dataset.element_spec)
  if keys is None:
    keys = list(shapes.keys())
  for k in keys:
    if k not in shapes:
      raise ValueError('Key %s not found in dataset.  Available keys are %s' %
                       (k, shapes.keys()))
    if not shapes[k].is_compatible_with(tf.TensorShape([None])):
      raise ValueError('Tensors to be packed must be one-dimensional.')
  # make sure that the length dictionary contains all keys as well as the
  # keys suffixed by "_segmentation" and "_position"
  if isinstance(key2length, int):
    key2length = {k: key2length for k in keys}
  for k in keys:
    for suffix in ['_segmentation', '_position']:
      key2length[k + suffix] = key2length[k]

  # trim to length
  dataset = dataset.map(
      lambda x: {k: x[k][:key2length[k]] for k in keys},
      num_parallel_calls=AUTOTUNE)
  # Setting batch_size=length ensures that the concatenated sequences (if they
  # have length >=1) are sufficient to fill at least one packed example.
  batch_size = max(key2length.values())
  dataset = dataset.padded_batch(
      batch_size, padded_shapes={k: [-1] for k in keys})
  dataset = _pack_with_tf_ops(dataset, keys, key2length)

  # Set the Tensor shapes correctly since they get lost in the process.
  def my_fn(x):
    return {k: tf.reshape(v, [key2length[k]]) for k, v in x.items()}

  return dataset.map(my_fn, num_parallel_calls=AUTOTUNE)


def _pack_with_tf_ops(dataset: tf.data.Dataset, keys: List[str],
                      key2length: Dict[str, int]) -> tf.data.Dataset:
  """Helper-function for packing a dataset which has already been batched.

  Helper for pack_dataset()  Uses tf.while_loop.

  Args:
    dataset: a dataset containing padded batches of examples.
    keys: a list of strings
    key2length: an dict from feature-key to integer

  Returns:
    a dataset.
  """
  empty_example = {}
  for k in keys:
    empty_example[k] = tf.zeros([0], dtype=tf.int32)
    empty_example[k + '_position'] = tf.zeros([0], dtype=tf.int32)
  keys_etc = empty_example.keys()

  def write_packed_example(partial, outputs):
    new_partial = empty_example.copy()
    new_outputs = {}
    for k in keys_etc:
      new_outputs[k] = outputs[k].write(
          outputs[k].size(),
          tf.pad(partial[k], [[0, key2length[k] - tf.size(partial[k])]]))
    return new_partial, new_outputs

  def map_fn(x):
    """Internal function to flat_map over.

    Consumes a batch of input examples and produces a variable number of output
    examples.
    Args:
      x: a single example

    Returns:
      a tf.data.Dataset
    """
    partial = empty_example.copy()
    i = tf.zeros([], dtype=tf.int32)
    dynamic_batch_size = tf.shape(x[keys[0]])[0]
    outputs = {}
    for k in keys:
      outputs[k] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True, element_shape=[key2length[k]])
      outputs[k + '_position'] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True, element_shape=[key2length[k]])

    def body_fn(i, partial, outputs):
      """Body function for while_loop.

      Args:
        i: integer scalar
        partial: dictionary of Tensor (partially-constructed example)
        outputs: dictionary of TensorArray

      Returns:
        A triple containing the new values of the inputs.
      """
      can_append = True
      one_example = {}
      for k in keys:
        val = tf.cast(x[k][i], tf.int32)
        val = val[:tf.reduce_sum(tf.cast(tf.not_equal(val, 0), tf.int32))]
        one_example[k] = val
      for k in keys:
        can_append = tf.logical_and(
            can_append,
            tf.less_equal(
                tf.size(partial[k]) + tf.size(one_example[k]), key2length[k]))

      def false_fn():
        return write_packed_example(partial, outputs)

      def true_fn():
        return partial, outputs

      partial, outputs = tf.cond(can_append, true_fn, false_fn)
      new_partial = {}
      for k in keys:
        new_seq = one_example[k][:key2length[k]]
        new_seq_len = tf.size(new_seq)
        new_partial[k] = tf.concat([partial[k], new_seq], 0)
        new_partial[k + '_position'] = tf.concat(
            [partial[k + '_position'],
             tf.range(new_seq_len)], 0)
      partial = new_partial
      return i + 1, partial, outputs

    # For loop over all examples in the batch.
    i, partial, outputs = tf.while_loop(
        cond=lambda *_: True,
        body=body_fn,
        loop_vars=(i, partial, outputs),
        shape_invariants=(
            tf.TensorShape([]),
            {k: tf.TensorShape([None]) for k in keys_etc},
            {k: tf.TensorShape(None) for k in keys_etc},
        ),
        maximum_iterations=dynamic_batch_size)
    _, outputs = write_packed_example(partial, outputs)
    packed = {k: outputs[k].stack() for k in keys_etc}
    for k in keys:
      packed[k + '_segmentation'] = (
          tf.cumsum(
              tf.cast(tf.equal(packed[k + '_position'], 0), tf.int32), axis=1) *
          tf.cast(tf.not_equal(packed[k], 0), tf.int32))
    return packed

  dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
  return dataset.unbatch()


# -----------------------------------------------------------------------------
# Main dataset prep routines.
# -----------------------------------------------------------------------------
def preprocess_wmt_data(dataset,
                        shuffle: bool,
                        num_epochs: Optional[int] = 1,
                        pack_examples: bool = True,
                        shuffle_buffer_size: int = 1024,
                        max_length: int = 512,
                        batch_size: int = 256,
                        drop_remainder: bool = True,
                        prefetch_size: int = AUTOTUNE):
  """Shuffle and batch/pack the given dataset."""

  def length_filter(max_len):

    def filter_fn(x):
      source, target = x['inputs'], x['targets']
      l = tf.maximum(tf.shape(source)[0], tf.shape(target)[0])
      return tf.less(l, max_len + 1)

    return filter_fn

  if max_length > 0:
    dataset = dataset.filter(length_filter(max_length))

  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.repeat(num_epochs)

  if pack_examples:
    dataset = pack_dataset(dataset, max_length)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  else:  # simple (static-shape) padded batching
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes={'inputs': max_length, 'targets': max_length},
        padding_values={'inputs': 0, 'targets': 0},
        drop_remainder=drop_remainder)

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  return dataset


def get_wmt_datasets(config: ml_collections.ConfigDict,
                     *,
                     n_devices: int,
                     reverse_translation: bool = True,
                     shard_idx: int = 0,
                     shard_count: int = 1,
                     vocab_path: Optional[str] = None,
                     pack_examples: bool = True):
  """Load and return dataset of batched examples for use during training."""
  batch_size = config.per_device_batch_size * n_devices
  if vocab_path is None:
    vocab_path = os.path.expanduser('~/wmt_sentencepiece_model')

  train_data, eval_data, _ = raw_wmt_datasets(
      dataset_name=config.dataset_name,
      eval_dataset_name=config.eval_dataset_name,
      reverse_translation=reverse_translation,
      shard_idx=shard_idx,
      shard_count=shard_count)

  sp_tokenizer = tokenizer.load_or_train_tokenizer(
      train_data,
      vocab_path=vocab_path,
      vocab_size=config.vocab_size,
      max_corpus_chars=config.max_corpus_chars)
  train_data = train_data.map(
      tokenizer.TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE)
  eval_data = eval_data.map(
      tokenizer.TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE)

  train_batches = preprocess_wmt_data(
      train_data,
      shuffle=True,
      num_epochs=None,
      pack_examples=pack_examples,
      batch_size=batch_size,
      max_length=config.max_target_length)

  eval_batches = preprocess_wmt_data(
      eval_data,
      shuffle=False,
      pack_examples=False,
      batch_size=batch_size,
      max_length=config.max_eval_target_length)

  predict_batches = preprocess_wmt_data(
      eval_data,
      shuffle=False,
      pack_examples=False,
      batch_size=batch_size,
      max_length=config.max_predict_length,
      drop_remainder=False)

  return train_batches, eval_batches, predict_batches, sp_tokenizer
