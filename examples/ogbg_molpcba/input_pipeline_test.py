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


class InputPipelineTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.dataset_length = 20
    self.datasets = input_pipeline.get_dummy_datasets(
        self.dataset_length, batch=False)

  @parameterized.product(
      valid_batch_size=[2, 5, 12, 15],
  )
  def test_estimate_padding_budget_valid(self, valid_batch_size):
    budget = input_pipeline.estimate_padding_budget_for_batch_size(
        self.datasets['train'], valid_batch_size, num_estimation_graphs=1)
    self.assertEqual(budget.n_graph, valid_batch_size)

  @parameterized.parameters(
      valid_batch_size=[-1, 0, 1],
  )
  def test_estimate_padding_budget_invalid(self, invalid_batch_size):
    with self.assertRaises(ValueError):
      input_pipeline.estimate_padding_budget_for_batch_size(
          self.datasets['train'], invalid_batch_size, num_estimation_graphs=1)


if __name__ == '__main__':
  absltest.main()
