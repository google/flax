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
"""Tests for flax.examples.imagenet.imagenet_lib."""

import itertools
import os
import tempfile

from absl.testing import absltest

from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.util import tensor_util
import tensorflow_datasets as tfds

import imagenet_lib


# TODO(mohitreddy): Refactor logic in testing/benchmark.py to create a 
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


class ImageNetLibTest(absltest.TestCase):
  """Test cases for imagenet_lib."""

  def test_train_and_evaluate(self):
    """Tests training and evaluation code by running a single step with
       mocked data for ImageNet dataset.
    """
    # Get datasets testing-metadata directory.
    current_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = parent_dir + '/testing/datasets'

    # Create a temporary directory where tensorboard metrics are written.
    model_dir = tempfile.mkdtemp()

    with tfds.testing.mock_data(num_examples=8, data_dir=data_dir):
      imagenet_lib.train_and_evaluate(
          model_dir=model_dir, batch_size=8, num_epochs=1,
          learning_rate=0.1, momentum=0.9, cache=False, half_precision=False,
          num_train_and_eval_steps=1, disable_checkpointing=True)

    summary_values_dict = _parse_and_return_summary_values(path=model_dir)

    # Since the values could change due to stochasticity in input processing
    # functions, model definition and dataset shuffling.
    self.assertGreaterEqual(summary_values_dict['train_accuracy'], 0.0)
    self.assertGreaterEqual(summary_values_dict['train_loss'], 0.0)
    self.assertGreaterEqual(summary_values_dict['eval_accuracy'], 0.0)
    self.assertGreaterEqual(summary_values_dict['eval_loss'], 0.0)


if __name__ == '__main__':
  absltest.main()
