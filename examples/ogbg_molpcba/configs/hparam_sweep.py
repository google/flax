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

"""Defines a sweep for the hyperparameters for the GNN."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Optimizer.
  config.optimizer = 'adam'
  config.learning_rate = None

  # Training hyperparameters.
  config.batch_size = None
  config.num_train_steps = 500_000
  config.log_every_steps = 50
  config.eval_every_steps = 1_000
  config.checkpoint_every_steps = 10_000
  config.add_virtual_node = True
  config.add_undirected_edges = True
  config.add_self_loops = True

  # GNN hyperparameters.
  config.model = 'GraphConvNet'
  config.message_passing_steps = None
  config.latent_size = None
  config.dropout_rate = None
  config.num_mlp_layers = 2
  config.num_classes = 128
  config.skip_connections = True
  config.layer_norm = True

  return config


def get_hyper(hyper):
  return hyper.product([
      hyper.sweep('config.add_virtual_node', [True, False]),
      hyper.sweep('config.add_undirected_edges', [True, False]),
      hyper.sweep('config.add_self_loops', [True, False]),
      hyper.sweep('config.layer_norm', [True, False]),
      hyper.sweep('config.skip_connections', [True, False]),
      hyper.sweep('config.batch_size', [256]),
      hyper.sweep('config.message_passing_steps', [5]),
      hyper.sweep('config.latent_size', [256]),
      hyper.sweep('config.learning_rate', [1e-3]),
      hyper.sweep('config.num_mlp_layers', [2]),
      hyper.sweep('config.dropout_rate', [0.1]),
  ])
