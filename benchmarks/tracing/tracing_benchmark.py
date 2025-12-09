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
"""Benchmarks for Jax tracing flax examples."""
import sys
from types import ModuleType
from typing import Any

from absl import app
from absl import flags
from absl import logging
import google_benchmark
import jax

flags.DEFINE_string(
    "example",
    None,
    "Example to benchmark. If unset, use google_benchmark to benchmark all.",
)

flags.DEFINE_enum(
    "mode",
    "trace_and_lower",
    ["trace", "lower", "trace_and_lower"],
    "Measure trace, lower, or trace_and_lower.",
)

# pylint: disable=unused-import
from flax.examples.gemma.configs import default as gemma_config
from flax.examples.imagenet.configs import default as imagenet_config
from flax.examples.lm1b.configs import default as lm1b_config
from flax.examples.mnist.configs import default as mnist_config
from flax.examples.nlp_seq.configs import default as nlp_seq_config
from flax.examples.ogbg_molpcba.configs import default as ogbg_molpcba_config
from flax.examples.ppo.configs import default as ppo_config
from flax.examples.seq2seq.configs import default as seq2seq_config
from flax.examples.sst2.configs import default as sst2_config
from flax.examples.vae.configs import default as vae_config
from flax.examples.wmt.configs import default as wmt_config
from flax.benchmarks.tracing import gemma
from flax.benchmarks.tracing import imagenet
from flax.benchmarks.tracing import lm1b
from flax.benchmarks.tracing import mnist
from flax.benchmarks.tracing import nlp_seq
from flax.benchmarks.tracing import ogbg_molpcba
from flax.benchmarks.tracing import ppo
from flax.benchmarks.tracing import seq2seq
from flax.benchmarks.tracing import sst2
from flax.benchmarks.tracing import vae
from flax.benchmarks.tracing import wmt
# pylint: enable=unused-import


def clear_caches(state):
  state.pause_timing()
  jax.clear_caches()
  state.resume_timing()


def benchmark_tracing(
    module: ModuleType, config_module: ModuleType, state: Any
) -> None:
  """Benchmark for tracing a flax example."""
  config = config_module.get_config()  # pytype: disable=attribute-error
  apply_fn, args, kwargs = module.get_apply_fn_and_args(config)  # pytype: disable=attribute-error
  while state:
    if flags.FLAGS.mode == "trace" or flags.FLAGS.mode == "trace_and_lower":
      _ = apply_fn.trace(*args, **kwargs)
      clear_caches(state)


def benchmark_lowering(
    module: ModuleType,
    config_module: ModuleType,
    state: Any,
    platform: str = "tpu",
) -> None:
  """Benchmark for lowering a flax example."""
  config = config_module.get_config()  # pytype: disable=attribute-error
  apply_fn, args, kwargs = module.get_apply_fn_and_args(config)  # pytype: disable=attribute-error
  traced = apply_fn.trace(*args, **kwargs)
  while state:
    if flags.FLAGS.mode == "lower" or flags.FLAGS.mode == "trace_and_lower":
      _ = traced.lower(lowering_platforms=(platform,))
      clear_caches(state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_gemma_trace(state):
  benchmark_tracing(gemma, gemma_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_gemma_lower(state):
  benchmark_lowering(gemma, gemma_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_imagenet_trace(state):
  benchmark_tracing(imagenet, imagenet_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_imagenet_lower(state):
  benchmark_lowering(imagenet, imagenet_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_lm1b_trace(state):
  benchmark_tracing(lm1b, lm1b_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_lm1b_lower(state):
  benchmark_lowering(lm1b, lm1b_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_mnist_trace(state):
  benchmark_tracing(mnist, mnist_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_mnist_lower(state):
  benchmark_lowering(mnist, mnist_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_nlp_seq_trace(state):
  benchmark_tracing(nlp_seq, nlp_seq_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_nlp_seq_lower(state):
  benchmark_lowering(nlp_seq, nlp_seq_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_vae_trace(state):
  benchmark_tracing(vae, vae_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_vae_lower(state):
  benchmark_lowering(vae, vae_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_wmt_trace(state):
  benchmark_tracing(wmt, wmt_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_wmt_lower(state):
  benchmark_lowering(wmt, wmt_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_sst2_trace(state):
  benchmark_tracing(sst2, sst2_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_sst2_lower(state):
  benchmark_lowering(sst2, sst2_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_seq2seq_trace(state):
  benchmark_tracing(seq2seq, seq2seq_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_seq2seq_lower(state):
  benchmark_lowering(seq2seq, seq2seq_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_ogbg_molpcba_trace(state):
  benchmark_tracing(ogbg_molpcba, ogbg_molpcba_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_ogbg_molpcba_lower(state):
  benchmark_lowering(ogbg_molpcba, ogbg_molpcba_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_ppo_trace(state):
  benchmark_tracing(ppo, ppo_config, state)


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_ppo_lower(state):
  benchmark_lowering(ppo, ppo_config, state)


def main(argv):
  del argv

  if flags.FLAGS.mode == "lower":
    raise ValueError(
        "`--mode=lower` is not supported when profiling a single example."
    )

  module = globals()[flags.FLAGS.example]
  config = globals()[flags.FLAGS.example + "_config"].get_config()
  apply_fn, args, kwargs, *_ = module.get_apply_fn_and_args(config)
  traced = apply_fn.trace(*args, **kwargs)
  lowered = traced.lower(lowering_platforms=("tpu",))

  logging.info("lowered: %s", lowered.as_text("hlo"))


if __name__ == "__main__":
  flags.FLAGS(sys.argv)
  flags.FLAGS.mark_as_parsed()

  if flags.FLAGS.example is None:
    google_benchmark.main()
  else:
    app.run(main)
