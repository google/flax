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

"""Testing Hyperparameter configuration."""

from configs import default as default_lib


def get_config():
  """Get the testing hyperparameter configuration.

  This configures pipeline to run a single training and evaluation step.
  Also minimized the model to reduce compilation time.
  """
  # Override default configuration to avoid duplication of field definition.
  testing_config = default_lib.get_config()

  testing_config.init_batch_size = 1
  testing_config.batch_size = 1
  testing_config.num_epochs = 1

  testing_config.n_resnet = 1
  testing_config.n_feature = 1

  testing_config.num_train_steps = 1
  testing_config.num_eval_steps = 1

  return testing_config
