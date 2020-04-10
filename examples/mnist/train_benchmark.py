"""Benchmark for the MNIST example."""
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


class MnistBenchmark(Benchmark):
  """Benchmarks for the MNIST Flax example."""

  @flagsaver
  def test_cpu(self):
    """Run full training for MNIST CPU training."""
    model_dir = tempfile.mkdtemp()
    FLAGS.model_dir = model_dir

    start_time = time.time()
    train.main([])
    benchmark_time = time.time() - start_time
    summaries = self.read_summaries(model_dir)

    # Summaries contain all the information necessary for the regression
    # metrics.
    wall_time, _, eval_accuracy = zip(*summaries['eval_accuracy'])
    wall_time = np.array(wall_time)
    sec_per_epoch = np.mean(wall_time[1:] - wall_time[:-1])
    end_eval_accuracy = eval_accuracy[-1]

    # Assertions are deffered until the test finishes, so the metrics are
    # always reported and benchmark success is determined based on *all*
    # assertions.
    self.assertBetween(sec_per_epoch, 14., 16.)
    self.assertBetween(end_eval_accuracy, 0.98, 1.0)

    # Use the reporting API to report single or multiple metrics/extras.
    self.report_wall_time(benchmark_time)
    self.report_metrics({'sec_per_epoch': sec_per_epoch,
                         'accuracy': end_eval_accuracy})
    self.report_extra('description', 'CPU test for MNIST.')


if __name__ == '__main__':
  absltest.main()
