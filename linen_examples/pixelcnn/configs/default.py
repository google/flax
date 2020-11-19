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

"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # The initial learning rate.
  config.learning_rate = 0.001

  # Learning rate decay, applied each optimization step.
  config.lr_decay = 0.999995

  # Batch size to use for data-dependent initialization.
  config.init_batch_size = 16

  # Batch size for training.
  config.batch_size = 64

  # Number of training epochs.
  config.num_epochs = 200

  # Dropout rate.
  config.dropout_rate = 0.5

  # Number of resnet layers per block.
  config.n_resnet = 5

  # Number of features in each conv layer.
  config.n_feature = 160

  # Number of components in the output distribution.
  config.n_logistic_mix = 10

  # Exponential decay rate of the sum of previous model iterates during Polyak
  # averaging.
  config.polyak_decay = 0.9995

  # Batch size for sampling.
  config.sample_batch_size = 256
  # Random number generator seed for sampling.
  config.sample_rng_seed = 0

  # Integer for PRNG random seed.
  config.seed = 0

  return config
