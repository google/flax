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

"""Tests for flax.examples.ogbg_molpcba.models."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jraph

import models


class ModelsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rngs = {
        'params': jax.random.PRNGKey(0),
        'dropout': jax.random.PRNGKey(1),
    }
    n_node = jnp.arange(3, 11)
    n_edge = jnp.arange(4, 12)
    total_n_node = jnp.sum(n_node)
    total_n_edge = jnp.sum(n_edge)
    n_graph = n_node.shape[0]
    feature_dim = 10
    self.graphs = jraph.GraphsTuple(
        n_node=n_node,
        n_edge=n_edge,
        senders=jnp.zeros(total_n_edge, dtype=jnp.int32),
        receivers=jnp.ones(total_n_edge, dtype=jnp.int32),
        nodes=jnp.ones((total_n_node, feature_dim)),
        edges=jnp.zeros((total_n_edge, feature_dim)),
        globals=jnp.zeros((n_graph, feature_dim)),
    )

  @parameterized.product(
      dropout_rate=[0., 0.5, 1.], output_size=[50, 100], num_layers=[2])
  def test_mlp(self, dropout_rate, output_size, num_layers):
    # Input definition.
    nodes = self.graphs.nodes

    # Model definition.
    mlp = models.MLP(
        feature_sizes=[output_size] * num_layers,
        dropout_rate=dropout_rate,
        activation=lambda x: x,
        deterministic=False)
    nodes_after_mlp, _ = mlp.init_with_output(self.rngs, nodes)

    # Test that dropout actually worked.
    num_masked_entries = jnp.sum(nodes_after_mlp == 0)
    num_total_entries = jnp.size(nodes_after_mlp)
    self.assertLessEqual(num_masked_entries,
                         (dropout_rate + 0.05) * num_total_entries)
    self.assertLessEqual((dropout_rate - 0.05) * num_total_entries,
                         num_masked_entries)

    # Test the shape of the output.
    self.assertEqual(nodes_after_mlp.shape[-1], output_size)

  @parameterized.parameters(
      {
          'latent_size': 5,
          'output_globals_size': 15,
          'use_edge_model': True,
      }, {
          'latent_size': 5,
          'output_globals_size': 15,
          'use_edge_model': False,
      })
  def test_graph_net(self, latent_size: int, output_globals_size: int,
                     use_edge_model: bool):
    # Input definition.
    graphs = self.graphs
    num_nodes = jnp.sum(graphs.n_node)
    num_edges = jnp.sum(graphs.n_edge)
    num_graphs = graphs.n_node.shape[0]

    # Model definition.
    net = models.GraphNet(
        latent_size=latent_size,
        num_mlp_layers=2,
        message_passing_steps=2,
        output_globals_size=output_globals_size,
        use_edge_model=use_edge_model)
    output, _ = net.init_with_output(self.rngs, graphs)

    # Output should be graph with the same topology, but a
    # different number of features.
    self.assertIsInstance(output, jraph.GraphsTuple)
    self.assertSequenceAlmostEqual(output.n_node, graphs.n_node)
    self.assertSequenceAlmostEqual(output.n_edge, graphs.n_edge)
    self.assertSequenceAlmostEqual(output.senders, graphs.senders)
    self.assertSequenceAlmostEqual(output.receivers, graphs.receivers)
    self.assertEqual(output.nodes.shape, (num_nodes, latent_size))
    self.assertEqual(output.edges.shape, (num_edges, latent_size))
    self.assertEqual(output.globals.shape, (num_graphs, output_globals_size))

  @parameterized.parameters({
      'latent_size': 15,
      'output_globals_size': 15
  }, {
      'latent_size': 5,
      'output_globals_size': 5
  })
  def test_graph_conv_net(self, latent_size: int, output_globals_size: int):
    graphs = self.graphs
    num_nodes = jnp.sum(graphs.n_node)
    num_graphs = graphs.n_node.shape[0]

    # Model definition.
    net = models.GraphConvNet(
        latent_size=latent_size,
        num_mlp_layers=2,
        message_passing_steps=2,
        output_globals_size=output_globals_size)
    output, _ = net.init_with_output(self.rngs, graphs)

    # Output should be graph with the same topology, but a
    # different number of features.
    self.assertIsInstance(output, jraph.GraphsTuple)
    self.assertSequenceAlmostEqual(output.n_node, graphs.n_node)
    self.assertSequenceAlmostEqual(output.n_edge, graphs.n_edge)
    self.assertSequenceAlmostEqual(output.edges.flatten(),
                                   graphs.edges.flatten())
    self.assertSequenceAlmostEqual(output.senders, graphs.senders)
    self.assertSequenceAlmostEqual(output.receivers, graphs.receivers)
    self.assertEqual(output.nodes.shape, (num_nodes, latent_size))
    self.assertEqual(output.globals.shape, (num_graphs, output_globals_size))


if __name__ == '__main__':
  absltest.main()
