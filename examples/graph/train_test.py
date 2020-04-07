# Lint as: python3

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

"""Tests for flax.examples.graph.train."""

from absl.testing import absltest
from train import GNN

from flax import nn
import jax
from jax import random
from jax import test_util as jtu
import jax.numpy as jnp

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class TrainTest(jtu.JaxTestCase):

  def test_permutation_invariance(self):

    num_nodes = 4
    num_features = 2
    rng = random.PRNGKey(0)

    # Generate random graph.
    adjacency = random.randint(rng, (num_nodes, num_nodes), 0, 2)
    node_feats = random.normal(rng, (num_nodes, num_features))
    sources, targets = jnp.where(adjacency)

    # Get permuted graph.
    perm = random.shuffle(rng, jnp.arange(num_nodes))
    node_feats_perm = node_feats[perm]
    adjacency_perm = adjacency[perm]
    for j in range(len(adjacency)):
      adjacency_perm = jax.ops.index_update(
          adjacency_perm, j, adjacency_perm[j][perm])
    sources_perm, targets_perm = jnp.where(adjacency_perm)

    # Create GNN.
    _, initial_params = GNN.init(
      rng, node_x=node_feats, edge_x=None, sources=sources, targets=targets)
    model = nn.Model(GNN, initial_params)

    # Feedforward both original and permuted graph.
    logits = model(node_feats, None, sources, targets)
    logits_perm = model(node_feats_perm, None, sources_perm, targets_perm)

    self.assertAllClose(logits[perm], logits_perm, check_dtypes=False)


if __name__ == '__main__':
  absltest.main()
