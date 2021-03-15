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

"""Cifar 10 input pipeline."""

import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds


class DataSource(object):
  """CIFAR10 data source."""

  TRAIN_IMAGES = 50000
  EVAL_IMAGES = 10000

  def __init__(self, config: ml_collections.ConfigDict, shuffle_seed: int = 1):

    dataset_builder = tfds.builder('cifar10')
    dataset_builder.download_and_prepare()
    self.ds_info = dataset_builder.info

    # Training set
    train_ds = dataset_builder.as_dataset(split='train').cache()
    train_ds = train_ds.repeat(config.num_epochs)
    train_ds = train_ds.shuffle(16 * config.batch_size, seed=shuffle_seed)

    def process_sample(x):
      image = tf.cast(x['image'], tf.float32)
      image = image / 127.5 - 1
      batch = {'image': image, 'label': x['label']}
      return batch

    train_ds = train_ds.map(process_sample, num_parallel_calls=128)
    train_ds = train_ds.batch(config.batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(10)
    self.train_ds = train_ds

    # Test set
    eval_ds = dataset_builder.as_dataset(split='test').cache()
    eval_ds = eval_ds.map(process_sample, num_parallel_calls=128)
    # Note: samples will be dropped if the number of test samples is not
    # divisible by the evaluation batch size
    eval_ds = eval_ds.batch(config.batch_size, drop_remainder=True)
    eval_ds = eval_ds.prefetch(10)
    self.eval_ds = eval_ds
