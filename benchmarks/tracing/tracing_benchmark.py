# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared benchmark utilities for Jax tracing flax examples."""

from collections.abc import Callable
import sys
from typing import Any

from absl import app
from absl import flags
from absl import logging
import google_benchmark
import jax

flags.DEFINE_enum(
    "mode",
    "trace_and_lower",
    ["trace", "lower", "trace_and_lower"],
    "Measure trace, lower, or trace_and_lower.",
)


def clear_caches(state):
  state.pause_timing()
  jax.clear_caches()
  state.resume_timing()


def benchmark_tracing(
    get_apply_fn_and_args: Callable[..., Any],
    get_config: Callable[[], Any],
    state: Any,
) -> None:
  """Benchmark for tracing a flax example."""
  config = get_config()
  apply_fn, args, kwargs = get_apply_fn_and_args(config)
  while state:
    if flags.FLAGS.mode == 'trace' or flags.FLAGS.mode == 'trace_and_lower':
      _ = apply_fn.trace(*args, **kwargs)
      clear_caches(state)


def benchmark_lowering(
    get_apply_fn_and_args: Callable[..., Any],
    get_config: Callable[[], Any],
    state: Any,
    platform: str = 'tpu',
) -> None:
  """Benchmark for lowering a flax example."""
  config = get_config()
  apply_fn, args, kwargs = get_apply_fn_and_args(config)
  traced = apply_fn.trace(*args, **kwargs)
  while state:
    if flags.FLAGS.mode == 'lower' or flags.FLAGS.mode == 'trace_and_lower':
      _ = traced.lower(lowering_platforms=(platform,))
      clear_caches(state)


def run_single_example(
    get_apply_fn_and_args: Callable[..., Any],
    get_config: Callable[[], Any],
) -> None:
  """Run a single example for profiling."""

  def main(argv):
    del argv
    if flags.FLAGS.mode == 'lower':
      raise ValueError(
          '`--mode=lower` is not supported when profiling a single example.'
      )
    config = get_config()
    apply_fn, args, kwargs, *_ = get_apply_fn_and_args(config)
    traced = apply_fn.trace(*args, **kwargs)
    lowered = traced.lower(lowering_platforms=('tpu',))
    logging.info('lowered: %s', lowered.as_text('hlo'))

  app.run(main)


def run_benchmarks() -> None:
  """Run registered google_benchmark benchmarks."""
  flags.FLAGS(sys.argv, known_only=True)
  flags.FLAGS.mark_as_parsed()
  google_benchmark.main()
