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

"""Tests for flax.examples.ogbg_molpcba.input_pipeline."""

from absl.testing import absltest
from absl.testing import parameterized
import input_pipeline
import jraph
import tensorflow as tf


def get_dummy_datasets(dataset_length: int):
  """Returns a set of datasets of unbatched GraphsTuples."""
  # The dummy graph.
  num_nodes = 3
  num_edges = 4
  dummy_graph = jraph.GraphsTuple(
      n_node=tf.expand_dims(num_nodes, 0),
      n_edge=tf.expand_dims(num_edges, 0),
      senders=tf.zeros(num_edges, dtype=tf.int32),
      receivers=tf.ones(num_edges, dtype=tf.int32),
      nodes=tf.zeros((num_nodes, 9)),
      edges=tf.ones((num_edges, 3)),
      globals=tf.ones((1, 128), dtype=tf.int64),
  )
  graphs_spec = input_pipeline.specs_from_graphs_tuple(dummy_graph)

  # Yields a set of graphs for the current split.
  def get_dummy_graphs():
    for _ in range(dataset_length):
      yield dummy_graph

  datasets = {}
  for split in ['train', 'validation', 'test']:
    datasets[split] = tf.data.Dataset.from_generator(
        get_dummy_graphs, output_signature=graphs_spec)
  return datasets


class InputPipelineTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    dataset_length = 20
    self.datasets = get_dummy_datasets(dataset_length)

  @parameterized.product(
      valid_batch_size=[2, 5, 12, 15],
  )
  def test_estimate_padding_budget_valid(self, valid_batch_size):
    budget = input_pipeline.estimate_padding_budget_for_batch_size(
        self.datasets['train'], valid_batch_size, num_estimation_graphs=1)
    self.assertEqual(budget.n_graph, valid_batch_size)

  @parameterized.product(
      invalid_batch_size=[-1, 0, 1],
  )
  def test_estimate_padding_budget_invalid(self, invalid_batch_size):
    with self.assertRaises(ValueError):
      input_pipeline.estimate_padding_budget_for_batch_size(
          self.datasets['train'], invalid_batch_size, num_estimation_graphs=1)


if __name__ == '__main__':
  absltest.main()
