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

"""Benchmark for the CIFAR10 example."""
import tempfile

import time
from absl import flags
from absl.testing import absltest
from absl.testing.flagsaver import flagsaver

import jax
import numpy as np
from flax.testing import Benchmark

import train


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


FLAGS = flags.FLAGS


class CifarTenBenchmark(Benchmark):
  """Benchmarks for the CIFAR10 Flax example."""

  @flagsaver
  def test_1x_v100(self):
    """Run Wide ResNet CIFAR10 on 1x V100 GPUs for 2 epochs."""
    model_dir = tempfile.mkdtemp()
    FLAGS.num_epochs = 2
    FLAGS.arch = 'wrn26_10'
    FLAGS.model_dir = model_dir

    start_time = time.time()
    train.main([])
    benchmark_time = time.time() - start_time
    summaries = self.read_summaries(model_dir)

    # Summaries contain all the information necessary for the regression
    # metrics.
    wall_time, _, eval_error_rate = zip(*summaries['eval_error_rate'])
    wall_time = np.array(wall_time)
    sec_per_epoch = np.mean(wall_time[1:] - wall_time[:-1])
    end_error_rate = eval_error_rate[-1]

    # Assertions are deferred until the test finishes, so the metrics are
    # always reported and benchmark success is determined based on *all*
    # assertions.
    self.assertBetween(sec_per_epoch, 80., 84.)
    self.assertBetween(end_error_rate, 0.30, 0.36)

    # Use the reporting API to report single or multiple metrics/extras.
    self.report_wall_time(benchmark_time)
    self.report_metrics({'sec_per_epoch': sec_per_epoch,
                         'error_rate': end_error_rate})
    self.report_extra(
        'description', 'Toy 1 x V100 test for CIFAR10 WideResNet26_10.')


if __name__ == '__main__':
  absltest.main()
