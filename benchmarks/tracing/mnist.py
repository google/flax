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
"""MNIST helper functions."""

from functools import partial
from typing import Any

from flax import nnx
from flax.benchmarks.tracing import tracing_benchmark
from flax.examples.mnist.configs import default as mnist_config
import google_benchmark
import jax
import jax.numpy as jnp
import ml_collections
import optax


class CNN(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
    self.batch_norm1 = nnx.BatchNorm(32, rngs=rngs)
    self.dropout1 = nnx.Dropout(rate=0.025)
    self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
    self.batch_norm2 = nnx.BatchNorm(64, rngs=rngs)
    self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
    self.dropout2 = nnx.Dropout(rate=0.025)
    self.linear2 = nnx.Linear(256, 10, rngs=rngs)

  def __call__(self, x, rngs: nnx.Rngs):
    x = self.avg_pool(nnx.relu(self.batch_norm1(self.dropout1(self.conv1(x), rngs=rngs))))
    x = self.avg_pool(nnx.relu(self.batch_norm2(self.conv2(x))))
    x = x.reshape(x.shape[0], -1)  # flatten
    x = nnx.relu(self.dropout2(self.linear1(x), rngs=rngs))
    x = self.linear2(x)
    return x


def loss_fn(model: CNN, batch, rngs):
  logits = model(batch['image'], rngs)
  loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label']
  ).mean()
  return loss, logits


def get_fake_batch(batch_size: int) -> dict[str, Any]:
  rng = jax.random.PRNGKey(0)
  images = jax.random.normal(rng, (batch_size, 28, 28, 1), jnp.float32)
  labels = jax.random.randint(rng, (batch_size,), 0, 10, jnp.int32)
  return {'image': images, 'label': labels}


def get_apply_fn_and_args(
    config: ml_collections.ConfigDict,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
  model = CNN(rngs=nnx.Rngs(0))
  batch = get_fake_batch(config.batch_size)
  rngs = nnx.Rngs(0)
  loss_fn_jit = jax.jit(loss_fn)
  return (
      loss_fn_jit,
      (model, batch, rngs),
      dict(),
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_mnist_trace(state):
  tracing_benchmark.benchmark_tracing(
      get_apply_fn_and_args, mnist_config.get_config, state
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_mnist_lower(state):
  tracing_benchmark.benchmark_lowering(
      get_apply_fn_and_args, mnist_config.get_config, state
  )


if __name__ == '__main__':
  tracing_benchmark.run_benchmarks()
