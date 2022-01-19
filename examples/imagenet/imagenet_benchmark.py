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

"""Benchmark for the ImageNet example."""

import time

from absl import flags
from absl.testing import absltest
from absl.testing.flagsaver import flagsaver
from flax.testing import Benchmark
import jax
import numpy as np

# Local imports.
import main
from configs import v100_x8_mixed_precision as config_lib


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

FLAGS = flags.FLAGS


class ImagenetBenchmark(Benchmark):
  """Benchmarks for the ImageNet Flax example."""

  @flagsaver
  def _test_8x_v100_half_precision(self, num_epochs: int, min_accuracy,
                                   max_accuracy):
    """Utility to benchmark ImageNet on 8xV100 GPUs. Use in your test func."""
    # Prepare and set flags defined in main.py.
    config = config_lib.get_config()
    config.num_epochs = num_epochs
    workdir = self.get_tmp_model_dir()

    FLAGS.workdir = workdir
    FLAGS.config = config

    start_time = time.time()
    main.main([])
    benchmark_time = time.time() - start_time
    summaries = self.read_summaries(workdir)

    # Summaries contain all the information necessary for the regression
    # metrics.
    wall_time, _, eval_accuracy = zip(*summaries['eval_accuracy'])
    wall_time = np.array(wall_time)
    sec_per_epoch = np.mean(wall_time[1:] - wall_time[:-1])
    end_accuracy = eval_accuracy[-1]

    # Assertions are deferred until the test finishes, so the metrics are
    # always reported and benchmark success is determined based on *all*
    # assertions.
    self.assertBetween(end_accuracy, min_accuracy, max_accuracy)

    # Use the reporting API to report single or multiple metrics/extras.
    self.report_wall_time(benchmark_time)
    self.report_metrics({'sec_per_epoch': sec_per_epoch,
                         'accuracy': end_accuracy})

  def test_8x_v100_half_precision_short(self):
    """Run ImageNet on 8x V100 GPUs in half precision for 2 epochs."""
    self._test_8x_v100_half_precision(
        num_epochs=2, min_accuracy=0.06, max_accuracy=0.09)
    self.report_extras({
        'description': 'Short (2 epochs) 8 x V100 test for ImageNet ResNet50.',
        'model_name': 'resnet50',
        'parameters': 'hp=true,bs=2048,num_epochs=2',
        'implementation': 'linen',
    })

  def test_8x_v100_half_precision_full(self):
    """Run ImageNet on 8x V100 GPUs in half precision for full 90 epochs."""
    self._test_8x_v100_half_precision(
        num_epochs=90, min_accuracy=0.76, max_accuracy=0.77)
    self.report_extras({
        'description': 'Full (90 epochs) 8 x V100 test for ImageNet ResNet50.',
        'model_name': 'resnet50',
        'parameters': 'hp=true,bs=2048,num_epochs=90',
        'implementation': 'linen',
    })


if __name__ == '__main__':
  absltest.main()
