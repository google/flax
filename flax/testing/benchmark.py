# Lint as: python3

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark class for Flax regression and integration testing.

This file defines utility functions for collecting model training results
from TensorBoard summaries, and for reporting benchmarks in a JSON format
for pickup by continuous integration / monitoring frameworks.

When the `benchmark_output_dir` is provided, the benchmark results are
saved in this directory in the JSON format with a single file per
benchmark.
"""

import functools
import itertools
import inspect
import json
from typing import Dict
import os
from absl import flags
from absl import logging
from absl.testing import absltest

from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_file_loader


flags.DEFINE_string(
  'benchmark_output_dir', default=None, help='Benchmark output directory.')


FLAGS = flags.FLAGS


def _make_events_generator(path):
  """Makes a generator yielding TensorBoard events from files in `path`."""
  return directory_watcher.DirectoryWatcher(
      path,
      event_file_loader.EventFileLoader).Load()


def _process_event(event):
  """Parse TensorBoard scalars into a (tag, wall_time, step, scalar) tuple."""
  for value in event.summary.value:
    if value.HasField('simple_value'):
      yield (value.tag, event.wall_time, event.step, value.simple_value)


def _get_tensorboard_scalars(path):
  """Read and parse scalar TensorBoard summaries.

  Args:
    path: str. Path containing TensorBoard event files.

  Returns:
    Dictionary mapping summary tags (str) to lists of
    (wall_time, step, scalar) tuples.
  """
  gen = _make_events_generator(path)
  data = filter(lambda x: x.HasField('summary'), gen)
  data = itertools.chain.from_iterable(map(_process_event, data))

  data_by_key = {}
  for tag, wall_time, step, value in data:
    if not tag in data_by_key:
      data_by_key[tag] = []
    data_by_key[tag].append((wall_time, step, value))
  return data_by_key


class Benchmark(absltest.TestCase):
  """Benchmark class for Flax examples.

  This class overrides changes the behaviour of `self.assert*` methods to
  deferred instead of immediate failure. This allows for using absltest
  assert methods for checking benchmark target metrics. This is also
  necessary for correctly reporting benchmark results and determining its
  success.
  """

  def __init__(self, *args, **kwargs):
    """Wrap test methods in a try-except decorator to delay exceptions."""
    super(Benchmark, self).__init__(*args, **kwargs)
    for func_name in dir(self):
      if func_name.startswith('assert'):
        func = getattr(self, func_name)
        patched_func = functools.partial(
            self._collect_assert_wrapper, func=func)
        setattr(self, func_name, patched_func)

  # pylint: disable=invalid-name
  def _collect_assert_wrapper(self, *args, func=None, **kwargs):
    """Wrapper around assert methods that caputres and collects failures."""
    try:
      return func(*args, **kwargs)
    except self.failureException as ex:
      self._outstanding_fails.append(ex)

  def setUp(self):
    """Setup ran before each test."""
    super(Benchmark, self).setUp()
    self._reported_name = None
    self._reported_wall_time = None
    self._reported_metrics = {}
    self._reported_extras = {}
    self._outstanding_fails = []

  def tearDown(self):
    """Tear down after each test."""
    super(Benchmark, self).tearDown()
    self._report_benchmark_results()
    for message in self._outstanding_fails:
      raise self.failureException(message)

  def has_outstanding_fails(self):
    """Determine whether the benchmark failed, but the error is deferred."""
    return len(self._outstanding_fails) > 0

  def read_summaries(self, path):
    """Read TensorBoard summaries."""
    return _get_tensorboard_scalars(path)

  def report_wall_time(self, wall_time: float):
    """Report wall time for the benchmark."""
    self._update_reported_name()
    self._reported_wall_time = wall_time

  def report_metrics(self, metrics: Dict[str, float]):
    """Report metrics for the benchmark."""
    self._update_reported_name()
    self._reported_metrics.update(metrics)

  def report_metric(self, name: str, value: float):
    """Report a single metric for the benchmark."""
    self.report_metrics({name: value})

  def report_extras(self, extras: Dict[str, str]):
    """Report extras for the benchmark."""
    self._update_reported_name()
    self._reported_extras.update(extras)

  def report_extra(self, name: str, value: str):
    """Report a single extra for the benchmark."""
    self.report_extras({name: value})

  def _get_test_name(self, prefix='test_'):
    """Returns full name of test class and method calling report_benchmark.

    The name is based on the *outermost* Benchmark class in the class stack.
    Based on tensorflow/python/platform/benchmark.py

    Args:
      prefix: str. Prefix that the caller method must have.

    Returns:
      Resolved test name as `ClassName.test_name`.
    """
    # Find the caller method (outermost Benchmark class).
    stack = inspect.stack()
    calling_class, name = None, None
    for frame_info in stack[::-1]:
      # print(frame_info.frame, frame_info.function)
      f_locals = frame_info.frame.f_locals
      f_self = f_locals.get('self', None)
      if isinstance(f_self, Benchmark):
        name = frame_info.function
        if name.startswith(prefix):
          calling_class = f_self
          break
    if calling_class is None:
      raise ValueError('Unable to determine the calling Benchmark class.')

    # Prefix the name with the class name.
    class_name = type(calling_class).__name__
    name = "%s.%s" % (class_name, name)
    return name

  def _update_reported_name(self):
    """Record / update test name for the benchmark."""
    self._reported_name = self._reported_name or self._get_test_name()

  def _report_benchmark_results(self):
    """Produce benchmark results report.

    Results are reported as a JSON string with the following schema:
    ```
    {
      "name": <class.testMethod>
      "succeeded": true / false
      "wall_time": float (containing wall-time for the benchmark)
      "metrics": {
        "string" -> float map of other performance metrics
      }
      "extras": {
        "string" -> "string" map containing anything else of interest
      }
    }
    ```
    """
    name = self._reported_name
    if not name:
      raise ValueError('Unable to determine test name for reporting '
                       'benchmark results. Make sure you are using '
                       '`self.report_*` methods.')

    succeeded = not self.has_outstanding_fails()
    results = {'name': name,
               'succeeded': succeeded,
               'metrics': self._reported_metrics,
               'extras': self._reported_extras}
    if self._reported_wall_time is not None:
      results['wall_time'] = self._reported_wall_time
    if not succeeded:
      msg = '\n'.join([str(fail) for fail in self._outstanding_fails])
      results['extras']['failed_assertions'] = msg

    results_str = json.dumps(results)
    logging.info(results_str)
    print(results_str)

    # Maybe save results as a file for pickup by CI / monitornig frameworks.
    benchmark_output_dir = FLAGS.benchmark_output_dir
    if benchmark_output_dir:
      os.makedirs(benchmark_output_dir, exist_ok=True)
      filename = os.path.join(benchmark_output_dir, name + '.json')
      with open(filename, 'w') as fout:
        json.dump(results, fout)
