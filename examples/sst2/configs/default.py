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

"""Default hyperparameter configuration for SST-2."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.embedding_size = 300
  config.hidden_size = 256
  config.vocab_size = None
  config.output_size = 1

  config.vocab_path = 'vocab.txt'
  config.max_input_length = 60

  config.dropout_rate = 0.5
  config.word_dropout_rate = 0.1
  config.unk_idx = 1

  config.learning_rate = 0.1
  config.momentum = 0.9
  config.weight_decay = 3e-6

  config.batch_size = 64
  config.bucket_size = 8
  config.num_epochs = 10

  config.seed = 0

  return config
