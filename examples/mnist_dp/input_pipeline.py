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

"""Input pipeline for MNIST-DP example."""

from typing import Dict

import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds


def normalize_image(sample: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
  return {
      'image': tf.cast(sample['image'], tf.float32) / 255.,
      'label': sample['label'],
  }


def get_datasets(config: ml_collections.ConfigDict):
  """Load standard MNIST train and test datasets."""
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  datasets_by_split = {}
  for split in ['train', 'validation', 'test']:
    if split == 'train':
      ds_split = ds_builder.as_dataset(split='train[:80%]')
      ds_split = ds_split.repeat(config.num_epochs)
      ds_split = ds_split.shuffle(buffer_size=1000, seed=config.train_rng_seed,
                                  reshuffle_each_iteration=True)
    if split == 'validation':
      ds_split = ds_builder.as_dataset(split='train[80%:]')
    elif split == 'test':
      ds_split = ds_builder.as_dataset(split='test')
    ds_split = ds_split.map(normalize_image)
    ds_split = ds_split.batch(config.batch_size)
    ds_split = tfds.as_numpy(ds_split)
    datasets_by_split[split] = ds_split
  return datasets_by_split
