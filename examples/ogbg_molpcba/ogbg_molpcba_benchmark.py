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

"""Benchmark for the ogbg_molpcba example."""

import time

from absl import flags
from absl.testing import absltest
from flax.testing import Benchmark
import jax
import numpy as np

import main
from configs import default
from configs import test


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

FLAGS = flags.FLAGS


class OgbgMolpcbaBenchmark(Benchmark):
  """Benchmarks for the ogbg_molpcba Flax example."""

  def test_1x_v100(self):
    """Run training with default config for ogbg_molpcba on a v100 GPU."""
    workdir = self.get_tmp_model_dir()
    config = default.get_config()

    FLAGS.workdir = workdir
    FLAGS.config = config

    start_time = time.time()
    main.main([])
    benchmark_time = time.time() - start_time

    summaries = self.read_summaries(workdir)

    # Summaries contain all the information necessary for
    # the regression metrics.
    wall_time, _, test_accuracy = zip(*summaries['test_accuracy'])
    wall_time = np.array(wall_time)
    sec_per_epoch = np.mean(wall_time[1:] - wall_time[:-1])
    end_test_accuracy = test_accuracy[-1]

    _, _, test_aps = zip(*summaries['test_mean_average_precision'])
    end_test_mean_average_precision = test_aps[-1]

    _, _, validation_accuracy = zip(*summaries['validation_accuracy'])
    end_validation_accuracy = validation_accuracy[-1]

    _, _, validation_aps = zip(*summaries['validation_mean_average_precision'])
    end_validation_mean_average_precision = validation_aps[-1]

    # Assertions are deferred until the test finishes, so the metrics are
    # always reported and benchmark success is determined based on *all*
    # assertions.
    self.assertGreaterEqual(end_test_mean_average_precision, 0.24)
    self.assertGreaterEqual(end_validation_mean_average_precision, 0.25)

    # Use the reporting API to report single or multiple metrics/extras.
    self.report_wall_time(benchmark_time)
    self.report_metrics({
        'sec_per_epoch':
            sec_per_epoch,
        'test_accuracy':
            end_test_accuracy,
        'test_mean_average_precision':
            end_test_mean_average_precision,
        'validation_accuracy':
            end_validation_accuracy,
        'validation_mean_average_precision':
            end_validation_mean_average_precision,
    })
    self.report_extras({
        'model_name': 'Graph Convolutional Network',
        'description': 'GPU (1x V100) test for ogbg_molpcba.',
        'implementation': 'linen',
    })

  def test_cpu(self):
    """Run training with test config for ogbg_molpcba on CPU."""
    workdir = self.get_tmp_model_dir()
    config = test.get_config()

    FLAGS.workdir = workdir
    FLAGS.config = config

    start_time = time.time()
    main.main([])
    benchmark_time = time.time() - start_time

    summaries = self.read_summaries(workdir)

    # Summaries contain all the information necessary for
    # the regression metrics.
    wall_time, _, test_accuracy = zip(*summaries['test_accuracy'])
    wall_time = np.array(wall_time)
    sec_per_epoch = np.mean(wall_time[1:] - wall_time[:-1])
    end_test_accuracy = test_accuracy[-1]

    _, _, test_aps = zip(*summaries['test_mean_average_precision'])
    end_test_mean_average_precision = test_aps[-1]

    _, _, validation_accuracy = zip(*summaries['validation_accuracy'])
    end_validation_accuracy = validation_accuracy[-1]

    _, _, validation_aps = zip(*summaries['validation_mean_average_precision'])
    end_validation_mean_average_precision = validation_aps[-1]

    # Use the reporting API to report single or multiple metrics/extras.
    self.report_wall_time(benchmark_time)
    self.report_metrics({
        'sec_per_epoch':
            sec_per_epoch,
        'test_accuracy':
            end_test_accuracy,
        'test_mean_average_precision':
            end_test_mean_average_precision,
        'validation_accuracy':
            end_validation_accuracy,
        'validation_mean_average_precision':
            end_validation_mean_average_precision,
    })
    self.report_extras({
        'model_name': 'Graph Convolutional Network',
        'description': 'CPU test for ogbg_molpcba.',
        'implementation': 'linen',
    })


if __name__ == '__main__':
  absltest.main()
