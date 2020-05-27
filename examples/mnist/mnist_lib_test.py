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
"""Tests for flax.examples.mnist.mnist_lib."""

import itertools
import os
import tempfile

from absl.testing import absltest

import jax
from jax import numpy as jnp

import numpy as onp

from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.util import tensor_util
import tensorflow_datasets as tfds

import mnist_lib


# TODO(#290): Refactor logic in testing/benchmark.py to create a
# utility class for parsing event files and extracting scalar summaries.
def _process_event(event):
  for value in event.summary.value:
    yield value


def _parse_and_return_summary_values(path):
  """Parses event file in given `path` and returns scalar summaries logged."""
  tag_event_value_dict = {}
  event_file_generator = directory_watcher.DirectoryWatcher(
      path, event_file_loader.EventFileLoader).Load()
  event_values = itertools.chain.from_iterable(
      map(_process_event, event_file_generator))
  for value in event_values:
    tag_event_value_dict[value.tag] = tensor_util.make_ndarray(value.tensor)

  return tag_event_value_dict


class MnistLibTest(absltest.TestCase):
  """Test cases for mnist_lib."""

  def test_cnn(self):
    """Tests CNN module used as the trainable model."""
    rng = jax.random.PRNGKey(0)
    output, init_params = mnist_lib.CNN.init_by_shape(
        rng, [((5, 224, 224, 3), jnp.float32)])

    self.assertEqual((5, 10), output.shape)

    # TODO(mohitreddy): Consider creating a testing module which
    # gives a parameters overview including number of parameters.
    self.assertLen(init_params, 4)


  def test_train_and_evaluate(self):
    """Tests training and evaluation code by running a single step with
       mocked data for MNIST dataset.
    """
    # Get datasets testing-metadata directory.
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = parent_dir + '/testing/datasets'

    # Create a temporary directory where tensorboard metrics are written.
    model_dir = tempfile.mkdtemp()

    with tfds.testing.mock_data(num_examples=8, data_dir=data_dir):
      mnist_lib.train_and_evaluate(
          model_dir=model_dir, num_epochs=1, batch_size=8,
          learning_rate=0.1, momentum=0.9)

    summary_values_dict = _parse_and_return_summary_values(path=model_dir)

    # Since the values could change due to stochasticity in input processing
    # functions, model definition and dataset shuffling.
    self.assertTrue(onp.allclose(summary_values_dict['train_accuracy'], 0.0))
    self.assertTrue(onp.allclose(summary_values_dict['train_loss'], 2.452725))
    self.assertTrue(onp.allclose(summary_values_dict['eval_accuracy'], 0.25))
    self.assertTrue(onp.allclose(summary_values_dict['eval_loss'], 1.908085))


if __name__ == '__main__':
  absltest.main()
