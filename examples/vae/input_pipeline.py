# Copyright 2024 The Flax Authors.
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

"""Input pipeline for VAE dataset."""

import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds


def build_train_set(batch_size, ds_builder):
  """Builds train dataset."""

  train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
  train_ds = train_ds.map(prepare_image)
  train_ds = train_ds.cache()
  train_ds = train_ds.repeat()
  train_ds = train_ds.shuffle(50000)
  train_ds = train_ds.batch(batch_size)
  train_ds = iter(tfds.as_numpy(train_ds))
  return train_ds


def build_test_set(ds_builder):
  """Builds train dataset."""
  test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
  test_ds = test_ds.map(prepare_image).batch(10000)
  test_ds = jnp.array(list(test_ds)[0])
  test_ds = jax.device_put(test_ds)
  return test_ds


def prepare_image(x):
  x = tf.cast(x['image'], tf.float32)
  x = tf.reshape(x, (-1,))
  return x
