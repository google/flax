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

"""Benchmark for the MNIST example."""
import time
from absl.testing import absltest

import jax
import numpy as np

from configs import default as config_lib
import mnist_lib

from flax.testing import Benchmark


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class MnistBenchmark(Benchmark):
  """Benchmarks for the MNIST Flax example."""

  def test_cpu(self):
    """Run full training for MNIST CPU training."""
    model_dir = self.get_tmp_model_dir()
    start_time = time.time()
    mnist_lib.train_and_evaluate(
      config=config_lib.get_config(),model_dir=model_dir)
    benchmark_time = time.time() - start_time
    summaries = self.read_summaries(model_dir)

    # Summaries contain all the information necessary for the regression
    # metrics.
    wall_time, _, eval_accuracy = zip(*summaries['eval_accuracy'])
    wall_time = np.array(wall_time)
    sec_per_epoch = np.mean(wall_time[1:] - wall_time[:-1])
    end_eval_accuracy = eval_accuracy[-1]

    # Assertions are deferred until the test finishes, so the metrics are
    # always reported and benchmark success is determined based on *all*
    # assertions.
    self.assertBetween(end_eval_accuracy, 0.98, 1.0)

    # Use the reporting API to report single or multiple metrics/extras.
    self.report_wall_time(benchmark_time)
    self.report_metrics({
        'sec_per_epoch': sec_per_epoch,
        'accuracy': end_eval_accuracy,
    })
    self.report_extras({
        'model_name': 'MNIST',
        'description': 'CPU test for MNIST.'
    })


if __name__ == '__main__':
  absltest.main()
