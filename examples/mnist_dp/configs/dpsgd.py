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

"""DP-SGD hyperparameter configuration."""

import ml_collections


def get_hyper(hyper):
  return hyper.product([
      hyper.sweep('config.noise_multiplier', [0.5, 1.0, 2.0]),
  ])


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.num_training_nodes = 48_000
  config.evaluation_cadence = 500
  config.train_rng_seed = 0
  config.differentially_private_training = True
  config.l2_norm_clip_percentile = 50
  config.noise_multiplier = 1.
  config.max_training_epsilon = 5
  config.dp_rng_seed = 1
  config.optimizer = 'sgd'
  config.learning_rate = 0.01
  config.momentum = 0.9
  config.nesterov = False
  config.batch_size = 200
  config.num_epochs = 100
  return config
