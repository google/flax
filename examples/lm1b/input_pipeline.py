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

"""Input pipeline for the lm1b dataset."""

from absl import logging
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

tf.enable_v2_behavior()


AUTOTUNE = tf.data.experimental.AUTOTUNE


def train_and_eval_dataset(dataset_name,
                           data_dir,
                           eval_holdout_size=0,
                           train_shuffle_files=True,
                           eval_shuffle_files=False):
  """Return train and evaluation datasets, feature info and supervised keys.

  Args:
    dataset_name: a string, the name of the TFDS dataset.
    data_dir: directory where the data is located.
    eval_holdout_size: float from 0 to <1; if >0 use this much of training data
      for evaluation (instead of looking for a pre-specified VALIDATION split).
    train_shuffle_files: Boolean determining whether or not to shuffle the train
      files at startup. Set to False if you want data determinism.
    eval_shuffle_files: Boolean determining whether or not to shuffle the test
      files at startup. Set to False if you want data determinism.

  Returns:
    a 4-tuple consisting of:
     * the train tf.Dataset
     * the eval tf.Dataset
     * information about features: a python dictionary with feature names
         as keys and an object as value that provides .shape and .n_classes.
     * supervised_keys: information what's the input and what's the target,
         ie., a pair of lists with input and target feature names.
  """
  dataset_builder = tfds.builder(dataset_name, data_dir=data_dir)
  info = dataset_builder.info
  splits = dataset_builder.info.splits
  if tfds.Split.TRAIN not in splits:
    raise ValueError("To train we require a train split in the dataset.")
  train_split = tfds.Split.TRAIN
  if eval_holdout_size > 0:
    holdout_percentage = int(eval_holdout_size * 100.0)
    train_percentage = 100 - holdout_percentage
    train_split = tfds.Split.TRAIN.subsplit(tfds.percent[:train_percentage])
    eval_split = tfds.Split.TRAIN.subsplit(tfds.percent[train_percentage:])
  else:
    if tfds.Split.VALIDATION not in splits and "test" not in splits:
      raise ValueError("We require a validation or test split in the dataset.")
    eval_split = tfds.Split.VALIDATION
    if tfds.Split.VALIDATION not in splits:
      eval_split = tfds.Split.TEST
  train = tfds.load(
      name=dataset_name,
      split=train_split,
      data_dir=data_dir,
      shuffle_files=train_shuffle_files)
  valid = tfds.load(
      name=dataset_name,
      split=eval_split,
      data_dir=data_dir,
      shuffle_files=eval_shuffle_files)
  keys = None
  if info.supervised_keys:
    keys = (info.supervised_keys[0], info.supervised_keys[1])
  return train, valid, info.features, keys


def bin_and_batch(dataset,
                  training,
                  n_devices,
                  target_batch_size=256,
                  target_bucket_length=32,
                  buckets=None,
                  max_eval_length=None,
                  drop_remainder=False):
  """Batching function, can specify batch size directly or per-device.

  Args:
    dataset: tf dataset containing individual sequences.
    training: bool: is this a train or eval dataset.
    n_devices: number of devices this dataset will be run on.
    target_batch_size: int: the target batch size for binned batches.
    target_bucket_length: int: the target sequence length for binned batches.
    buckets: (List[int], List[int]): manually specified length buckets and
      batch sizes for bins.
    max_eval_length: int: for eval set allow a extra long-sequence bin.
    drop_remainder: bool: if true drop last batch if not divisible by
      batch sizes. (e.g. not divisible by n_devices).

  Returns:
    Dynamically binned batches of sequence that roughly keep the total
    number of tokens (target_batch_size * target_bucket_length) the same, while
    insuring batch sizes are divisible by n_devices for distributed training.
  """
  # Create heuristic buckets is none are specified.
  if buckets is None:
    logging.info("Heuristically bucketing based on shapes of examples.")
    bucket_boundaries = [
        target_bucket_length // 4, target_bucket_length // 2,
        target_bucket_length, target_bucket_length * 2,
        target_bucket_length * 4, target_bucket_length * 8,
        target_bucket_length * 16
    ]
    bucket_batch_sizes = [
        target_batch_size * 4, target_batch_size * 2,
        target_batch_size, target_batch_size // 2,
        target_batch_size // 4, target_batch_size // 8,
        target_batch_size // 16
    ]
    # allow for different evaluation max-length bucket and batchsize
    if not training:
      max_eval_length = max_eval_length or target_bucket_length * 32
      bucket_boundaries[-1] = max_eval_length
      bucket_batch_sizes[-1] = target_batch_size // max_eval_length
    # We will pad to boundaries which pads to bucket_boundary-1: add 1 here.
    bucket_boundaries = [b + 1 for b in bucket_boundaries]
    # Make batch sizes divisible by n_devices.
    bucket_batch_sizes = [
        max(b // n_devices, 1) * n_devices for b in bucket_batch_sizes
    ]
    buckets = (bucket_boundaries, bucket_batch_sizes)

  logging.info("Bucketing with buckets %s.", str(buckets))

  def length_fn(sequence):
    """Returns length of sequence."""
    return tf.shape(sequence)[0]

  boundaries, batch_sizes = buckets
  # bucket_by_sequence_length expects a final dummy 1 batch_size
  batch_sizes.append(1)
  dataset = dataset.apply(
      tf.data.experimental.bucket_by_sequence_length(
          length_fn,
          boundaries,
          batch_sizes,
          pad_to_bucket_boundary=True,
          drop_remainder=drop_remainder))
  return dataset


def lm1b_preprocess(dataset,
                    training,
                    n_devices,
                    shuffle_buffer_batches=10,
                    max_target_length=512,
                    max_eval_target_length=2048,
                    batch_size=256,
                    buckets=None,
                    single_training_epoch=False,
                    drop_remainder=True,
                    prefetch_size=AUTOTUNE):
  """Shuffle and batch the given dataset.

  Args:
    dataset: tf dataset containing individual sequences.
    training: bool: is this a train or eval dataset.
    n_devices: number of devices this dataset will be run on.
    shuffle_buffer_batches: int: shuffle buffer as multiples of batch_size.
    max_target_length: int: drop training sequences longer than this.
    max_eval_target_length: int: drop evaluation sequences longer than this.
    batch_size: int: target batch size, if using dynamic binning the actual
      batch sizes will be small fractions or multiples of this target.
    buckets: (List[int], List[int]): manually specified length buckets and
      batch sizes for bins.
    single_training_epoch: bool: whether to produce only a single training
      epoch or to repeat training dataset indefinitely.
    drop_remainder: bool: if true drop last batch if not divisible by
      batch sizes. (e.g. not divisible by n_devices).
    prefetch_size: int or AUTOTUNE: number of batches to prefetch.

  Returns:
    Dynamically binned batches of sequence that roughly keep the total
    number of tokens the same, while insuring batch sizes are divisible by
    n_devices for distributed training.
  """

  # Filter dataset by training or evaluation length cutoffs.
  def length_filter(max_len):

    def filter_fn(source):
      return tf.less(tf.shape(source)[0], max_len + 1)

    return filter_fn

  if max_target_length > 0 and training:
    dataset = dataset.filter(length_filter(max_target_length))
  if max_eval_target_length > 0 and not training:
    dataset = dataset.filter(length_filter(max_eval_target_length))

  # Shuffle and repeat training set.
  if training:
    dataset = dataset.shuffle(shuffle_buffer_batches * batch_size)
    if not single_training_epoch:
      dataset = dataset.repeat()

  # Batch into padded, length-binned batches.
  dataset = bin_and_batch(
      dataset,
      training,
      n_devices,
      target_batch_size=batch_size,
      max_eval_length=max_eval_target_length,
      buckets=buckets,
      drop_remainder=drop_remainder)

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  return dataset


def get_lm1b_datasets(n_devices,
                      batch_size=256,
                      dynamic_batching=True,
                      max_target_length=512,
                      max_eval_target_length=2048,
                      drop_remainder=True,
                      single_training_epoch=False):
  """Load and return dataset of batched examples for use during training.

  Note that dynamic batching is not currently compatible with multihost
  training. For that application we need to pre-bin data and e.g. use a shared
  rng key across hosts in pmap to choose identical bin-shapes to satify the
  requirement for SPMD shape identity across multihost pmap.

  Args:
    n_devices: number of devices this dataset will be run on.
    batch_size: int: target batch size, if using dynamic binning the actual
      batch sizes will be small fractions or multiples of this target.
    dynamic_batching: bool: whether to use dynamic length-binning to produce
      batches with roughly constant token count.
    max_target_length: int: drop training sequences longer than this.
    max_eval_target_length: int: drop evaluation sequences longer than this.
    drop_remainder: bool: if true drop last batch if not divisible by
      batch sizes. (e.g. not divisible by n_devices).
    single_training_epoch: bool: whether to produce only a single training
      epoch or to repeat training dataset indefinitely.

  Returns:
    Tuple of:
    training tf dataset, evaluation tf dataset, and tfds features info
  """
  if batch_size % n_devices:
    raise ValueError("Batch size %d isn't divided evenly by n_devices %d" %
                     (batch_size, n_devices))

  (train_data, eval_data, features_info, keys) = train_and_eval_dataset(
      "lm1b/subwords32k",
      None,
      train_shuffle_files=True,
      eval_shuffle_files=False)

  # For sequence models, TFDS yields a (duplicated) feature dictionary
  # we want to simplify things by mapping e.g.
  # {key0: inputs_data, key1: targets_data} to just inputs_data
  # for LM1B: key0 == key1 == text and inputs_data == targets_data
  inputs_key, targets_key = keys
  del targets_key
  to_sequence_data = lambda x: x[inputs_key]
  train_data = train_data.map(to_sequence_data)
  eval_data = eval_data.map(to_sequence_data)

  if dynamic_batching:
    train_buckets = None
    eval_buckets = None
  else:
    # buckets should be (bucket_boundaries, bucket_batch_sizes)
    train_buckets = ([max_target_length + 1], [batch_size])
    eval_buckets = ([max_eval_target_length + 1], [batch_size])

  train_batches = lm1b_preprocess(
      train_data,
      training=True,
      n_devices=n_devices,
      batch_size=batch_size,
      max_target_length=max_target_length,
      max_eval_target_length=max_eval_target_length,
      buckets=train_buckets,
      single_training_epoch=single_training_epoch,
      drop_remainder=drop_remainder)

  eval_batches = lm1b_preprocess(
      eval_data,
      training=False,
      n_devices=n_devices,
      max_target_length=max_target_length,
      max_eval_target_length=max_eval_target_length,
      batch_size=batch_size,
      buckets=eval_buckets,
      drop_remainder=drop_remainder)

  return train_batches, eval_batches, features_info
