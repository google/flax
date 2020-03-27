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

"""Input pipeline for the wmt de-en dataset.

NOTES:
  - Not set up for dynamic batching with multihost training!
"""
import copy
import os
from absl import logging
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE


def save_tfds_subword_vocab(encoder, path):
  """Save tfds SubwordTextEncoder vocab to given path."""
  subword_data = '\n'.join(["'%s'" % s for s in encoder.subwords])
  with tf.io.gfile.GFile(path, 'w') as fp:
    fp.write(subword_data)


def load_tfds_subword_vocab(path):
  """Load tfds SubwordTextEncoder vocab from given path."""
  with tf.io.gfile.GFile(path, 'r') as fp:
    subword_data = fp.read()
  # We remove the quotes around each token in the tfds vocab save format.
  vocab_list = [line[1:-1] for line in subword_data.split('\n')]
  return tfds.features.text.SubwordTextEncoder(vocab_list=vocab_list)


def bin_and_batch(dataset,
                  n_devices,
                  batch_size=256,
                  bucket_length=32,
                  buckets=None,
                  drop_remainder=True):
  """Batching function."""
  # Create heuristic buckets is none are specified.
  if buckets is None:
    logging.info('Heuristically bucketing based on shapes of examples.')
    bucket_boundaries = [
        bucket_length // 4, bucket_length // 2, bucket_length,
        bucket_length * 2, bucket_length * 4, bucket_length * 8,
        bucket_length * 16
    ]
    bucket_batch_sizes = [
        batch_size * 4, batch_size * 2, batch_size,
        batch_size // 2, batch_size // 4, batch_size // 8,
        batch_size // 16
    ]
    # TF.data's bucket_by_sequence_length pads to (bucket_boundary - 1):
    # we add 1 here to pad to the correct specified length.
    bucket_boundaries = [b + 1 for b in bucket_boundaries]
    # Make batch sizes divisible by n_devices.
    bucket_batch_sizes = [
        max(b // n_devices, 1) * n_devices for b in bucket_batch_sizes
    ]
    buckets = (bucket_boundaries, bucket_batch_sizes)

  logging.info('Bucketing with buckets %s.', str(buckets))

  def example_length(example):
    """The length function used by bucket_by_sequence_length to bucket."""
    return tf.maximum(tf.shape(example['inputs'])[0],
                      tf.shape(example['targets'])[0])

  boundaries, batch_sizes = buckets
  # bucket_by_sequence_length expects a final dummy 1 batch_size.
  batch_sizes.append(1)
  dataset = dataset.apply(
      tf.data.experimental.bucket_by_sequence_length(
          example_length,
          boundaries,
          batch_sizes,
          pad_to_bucket_boundary=True,
          drop_remainder=drop_remainder))
  return dataset


def preprocess_wmt_data(dataset,
                        training,
                        n_devices,
                        shuffle_buffer_size=1024,
                        max_length=512,
                        batch_size=256,
                        bucket_length=32,
                        buckets=None,
                        drop_remainder=True,
                        prefetch_size=AUTOTUNE):
  """Shuffle and batch the given dataset."""
  buckets = copy.copy(buckets)

  def length_filter(max_len):
    def filter_fn(x):
      source, target = x['inputs'], x['targets']
      l = tf.maximum(tf.shape(source)[0], tf.shape(target)[0])
      return tf.less(l, max_len + 1)
    return filter_fn

  if max_length > 0:
    dataset = dataset.filter(length_filter(max_length))

  if training:
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat()

  dataset = bin_and_batch(
      dataset,
      n_devices,
      batch_size=batch_size,
      bucket_length=bucket_length,
      drop_remainder=drop_remainder,
      buckets=buckets)

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  return dataset


def get_wmt_datasets(n_devices,
                     dataset_name='wmt17_translate/de-en',
                     data_dir=None,
                     vocab_path=None,
                     target_vocab_size=2**15,
                     max_corpus_chars=10**7,
                     batch_size=256,
                     bucket_length=32,
                     dynamic_batching=True,
                     max_target_length=512,
                     max_eval_target_length=512):
  """Load and return dataset of batched examples for use during training."""
  if batch_size % n_devices:
    raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                     (batch_size, n_devices))
  if vocab_path is None:
    vocab_path = os.path.expanduser('~/vocab.subwords')

  train_data = tfds.load(
      name=dataset_name,
      split=tfds.Split.TRAIN,
      data_dir=data_dir,
      shuffle_files=True)

  eval_data = tfds.load(
      name=dataset_name,
      split=tfds.Split.VALIDATION,
      data_dir=data_dir,
      shuffle_files=False)
  features_info = tfds.builder(dataset_name, data_dir=data_dir).info

  # standardize on 'inputs' and 'targets' features.
  # e.g. {'de': example_in, 'en': example_out} to
  #      {'inputs': example_in, 'targets': example_out}
  to_features_dict = (
      lambda x: {'inputs': x[features_info.supervised_keys[0]],
                 'targets': x[features_info.supervised_keys[1]]})
  train_data = train_data.map(to_features_dict)
  eval_data = eval_data.map(to_features_dict)

  try:
    subword_tokenizer = load_tfds_subword_vocab(vocab_path)
  except tf.errors.NotFoundError:
    logging.info('Subword vocab not found, building shared subword dictionary.')
    def corpus_generator():
      for data in train_data:
        yield data['inputs'].numpy()
        yield data['targets'].numpy()
    subword_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        corpus_generator(),
        target_vocab_size=target_vocab_size,
        max_corpus_chars=max_corpus_chars)
    # save vocab
    save_tfds_subword_vocab(subword_tokenizer, vocab_path)

  # Encode strings with subword tokenizer.
  # NOTE(levskaya): tokenization is currently an unacceptably slow part of
  # the input pipeline, we'll be factoring out dataset preparation from
  # training pipeline in another PR soon.
  def encode(lang1, lang2):
    res1 = subword_tokenizer.encode(
        lang1.numpy()) + [subword_tokenizer.vocab_size]  # Add EOS Token
    res2 = subword_tokenizer.encode(
        lang2.numpy()) + [subword_tokenizer.vocab_size]  # Add EOS Token
    return res1, res2

  def tf_encode(x):
    src, tgt = x['inputs'], x['targets']
    result_src, result_tgt = tf.py_function(encode,
                                            [src, tgt],
                                            [tf.int64, tf.int64])
    result_src.set_shape([None])
    result_tgt.set_shape([None])
    return {'inputs': result_src, 'targets': result_tgt}

  train_data = train_data.map(tf_encode, num_parallel_calls=AUTOTUNE)
  eval_data = eval_data.map(tf_encode, num_parallel_calls=AUTOTUNE)

  # Set bucketing.
  if dynamic_batching:
    train_buckets = None
    eval_buckets = None
  else:
    # Buckets should be (bucket_boundaries, bucket_batch_sizes). For static
    # batching, use a single bucket boundary and bucket batch size.
    train_buckets = ([max_target_length + 1], [batch_size])
    eval_buckets = ([max_eval_target_length + 1], [batch_size])

  train_batches = preprocess_wmt_data(
      train_data,
      training=True,
      n_devices=n_devices,
      batch_size=batch_size,
      bucket_length=bucket_length,
      max_length=max_target_length,
      buckets=train_buckets)

  eval_batches = preprocess_wmt_data(
      eval_data,
      training=False,
      n_devices=n_devices,
      batch_size=batch_size,
      bucket_length=bucket_length,
      max_length=max_eval_target_length,
      buckets=eval_buckets)

  predict_batches = preprocess_wmt_data(
      eval_data,
      training=False,
      n_devices=n_devices,
      batch_size=batch_size,
      bucket_length=bucket_length,
      max_length=max_eval_target_length,
      buckets=eval_buckets,
      drop_remainder=False)

  return train_batches, eval_batches, predict_batches, subword_tokenizer
