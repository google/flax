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

"""Hyperparameter configuration for Fake data benchmark."""

import jax

from configs import default as default_lib


def get_config():
  """Get the hyperparameter configuration for Fake data benchmark."""
  # Override default configuration to avoid duplication of field definition.
  config = default_lib.get_config()
  config.batch_size = 256 * jax.device_count()
  config.half_precision = True
  config.num_epochs = 5

  # Previously the input pipeline computed:
  # `steps_per_epoch` as input_pipeline.TRAIN_IMAGES // batch_size
  config.num_train_steps = 1024 // config.batch_size
  # and `steps_per_eval` as input_pipeline.EVAL_IMAGES // batch_size
  config.steps_per_eval = 512 // config.batch_size

  return config
