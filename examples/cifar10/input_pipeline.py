# Copyright 2020 The Flax Authors.
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

"""CIFAR-10 input pipeline."""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3

# Computed ourselves
MEAN_RGB = [0.4914 * 255, 0.4822 * 255, 0.4465 * 255]
STDDEV_RGB = [0.2470 * 255, 0.2435 * 255, 0.2616 * 255]


def augment(image, crop_padding=4, flip_lr=True):
  """Augment small image with random crop and h-flip.

  Args:
    image: image to augment
    crop_padding: random crop range
    flip_lr: if True perform random horizontal flip

  Returns:
    augmented image
  """
  assert crop_padding >= 0
  if crop_padding > 0:
    # Pad with reflection padding
    # (See https://arxiv.org/abs/1605.07146)
    # Section 3
    image = tf.pad(
        image, [[crop_padding, crop_padding],
                [crop_padding, crop_padding], [0, 0]], 'REFLECT')

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

  if flip_lr:
    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  return image


class CIFAR10DataSource(object):
  """CIFAR-10 data source."""
  TRAIN_IMAGES = 50000
  EVAL_IMAGES = 10000

  # Computed from the training set by taking the per-channel mean/std-dev
  # over sample, height and width axes of all training samples.
  MEAN_RGB = [0.4914 * 255, 0.4822 * 255, 0.4465 * 255]
  STDDEV_RGB = [0.2470 * 255, 0.2435 * 255, 0.2616 * 255]

  CAN_FLIP_HORIZONTALLY = True

  def __init__(self, train_batch_size, eval_batch_size, shuffle_seed=1):
    mean_rgb = tf.constant(self.MEAN_RGB, shape=[1, 1, 3], dtype=tf.float32)
    std_rgb = tf.constant(self.STDDEV_RGB, shape=[1, 1, 3], dtype=tf.float32)
    if self.CAN_FLIP_HORIZONTALLY is None:
      raise ValueError
    flip_lr = self.CAN_FLIP_HORIZONTALLY

    # Training set
    train_ds = tfds.load('cifar10', split='train').cache()
    train_ds = train_ds.repeat()
    train_ds = train_ds.shuffle(16 * train_batch_size, seed=shuffle_seed)

    def _process_train_sample(x):
      image = tf.cast(x['image'], tf.float32)
      image = augment(image, crop_padding=4, flip_lr=flip_lr)
      image = (image - mean_rgb) / std_rgb
      batch = {'image': image, 'label': x['label']}
      return batch

    train_ds = train_ds.map(_process_train_sample, num_parallel_calls=128)
    train_ds = train_ds.batch(train_batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(10)
    self.train_ds = train_ds

    # Test set
    eval_ds = tfds.load('cifar10', split='test').cache()

    def _process_test_sample(x):
      image = tf.cast(x['image'], tf.float32)
      image = (image - mean_rgb) / std_rgb
      batch = {'image': image, 'label': x['label']}
      return batch

    eval_ds = eval_ds.map(_process_test_sample, num_parallel_calls=128)
    # Note: samples will be dropped if the number of test samples
    # (EVAL_IMAGES=10000) is not divisible by the evaluation batch
    # size
    eval_ds = eval_ds.batch(eval_batch_size, drop_remainder=True)
    eval_ds = eval_ds.repeat()
    eval_ds = eval_ds.prefetch(10)
    self.eval_ds = eval_ds
