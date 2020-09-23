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

  config.learning_rate = 0.001
  config.lr_decay = 0.999995
  config.init_batch_size = 16
  config.batch_size = 64
  config.num_epochs = 200
  config.dropout_rate = 0.5

  config.rng = 0

  config.n_resnet = 5
  config.n_feature = 160
  config.n_logistic_mix = 10

  config.polyak_decay = 0.9995

  config.num_train_steps = -1
  config.num_eval_steps = -1

  return config
