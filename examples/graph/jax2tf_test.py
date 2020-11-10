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

# Lint as: python3
"""Jax2Tf tests for flax.examples.graph.train."""

from absl.testing import absltest

from flax import nn
from flax import optim
import train
from train import GNN
from flax.testing import jax2tf_test_util

import jax
from jax import random
from jax import test_util as jtu
from jax.experimental import jax2tf

import numpy as np

DEFAULT_ATOL = 1e-6


def _create_model(node_feats, sources, targets):
  prng = random.PRNGKey(0)
  _, initial_params = GNN.init(
      prng, node_x=node_feats, edge_x=None, sources=sources, targets=targets)
  return nn.Model(GNN, initial_params)


def _single_train_step(node_feats, sources, targets):
  model = _create_model(node_feats, sources, targets)
  optimizer = optim.Adam(learning_rate=0.01).create(model)
  _, loss = train.train_step(
      optimizer=optimizer,
      node_feats=node_feats,
      sources=sources,
      targets=targets)
  return loss


def _eval(node_feats, node_labels, sources, targets):
  model = _create_model(node_feats, sources, targets)
  return train.eval_step(model, node_feats, sources, targets, node_labels)


class Jax2TfTest(jax2tf_test_util.JaxToTfTestCase):
  """Tests that compare the results of model w/ and w/o using jax2tf."""

  def test_single_train_step(self):
    #       0 (0)
    #      / \
    # (0) 1 - 2 (1)
    edge_list = [(0, 0), (1, 2), (2, 0)]
    node_labels = [0, 0, 1]
    node_feats, node_labels, sources, targets = train.create_graph_data(
        edge_list=edge_list, node_labels=node_labels)
    np.testing.assert_allclose(
        _single_train_step(node_feats, sources, targets),
        jax2tf.convert(_single_train_step)(node_feats, sources, targets),
        atol=DEFAULT_ATOL)

  def test_eval(self):
    # Get Zachary's karate club graph dataset.
    node_feats, node_labels, sources, targets = train.get_karate_club_data()
    np.testing.assert_allclose(
        _eval(node_feats, node_labels, sources, targets),
        jax2tf.convert(_eval)(node_feats, node_labels, sources, targets),
        atol=DEFAULT_ATOL)


if __name__ == '__main__':
  # Parse absl flags test_srcdir and test_tmpdir.
  jax.config.parse_flags_with_absl()
  absltest.main(testLoader=jtu.JaxTestLoader())
