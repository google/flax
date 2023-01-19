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

"""Benchmark for the ImageNet example using fake data for quick perf results.

This script doesn't need the dataset, but it needs the dataset metadata.
That can be fetched with the script `flax/tests/download_dataset_metadata.sh`.
"""

import pathlib
import time

from absl.testing import absltest
from absl.testing.flagsaver import flagsaver
from flax.testing import Benchmark
import jax
import tensorflow_datasets as tfds

# Local imports.
from configs import fake_data_benchmark as config_lib
import train

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class ImagenetBenchmarkFakeData(Benchmark):
  """Runs ImageNet using fake data for quickly measuring performance."""

  def test_fake_data(self):
    workdir = self.get_tmp_model_dir()
    config = config_lib.get_config()
    # Go two directories up to the root of the flax directory.
    flax_root_dir = pathlib.Path(__file__).absolute().parents[2]
    data_dir = str(flax_root_dir) + '/.tfds/metadata'

    # Warm-up first so that we are not measuring just compilation.
    with tfds.testing.mock_data(num_examples=1024, data_dir=data_dir):
      train.train_and_evaluate(config, workdir)

    start_time = time.time()
    with tfds.testing.mock_data(num_examples=1024, data_dir=data_dir):
      train.train_and_evaluate(config, workdir)
    benchmark_time = time.time() - start_time

    self.report_wall_time(benchmark_time)
    self.report_extras({
        'description': 'ImageNet ResNet50 with fake data',
        'model_name': 'resnet50',
        'parameters': f'hp=true,bs={config.batch_size}',
    })


if __name__ == '__main__':
  absltest.main()
