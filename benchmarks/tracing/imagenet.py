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
"""ImageNet helper functions for benchmarking."""

import functools
from typing import Any

from flax import linen as nn
from flax.benchmarks.tracing import tracing_benchmark
from flax.examples.imagenet import models
from flax.examples.imagenet.configs import default as imagenet_config
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import train_state
import google_benchmark
import jax
import jax.numpy as jnp
import ml_collections
import optax

NUM_CLASSES = 1000


class TrainState(train_state.TrainState):
  batch_stats: Any
  dynamic_scale: dynamic_scale_lib.DynamicScale


def create_model(*, model_cls, half_precision, **kwargs):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32
  return model_cls(num_classes=NUM_CLASSES, dtype=model_dtype, **kwargs)


def initialized(key, image_size, model):
  input_shape = (1, image_size, image_size, 3)

  @jax.jit
  def init(*args):
    return model.init(*args)

  variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
  return variables['params'], variables['batch_stats']


def cross_entropy_loss(logits, labels):
  one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
  return jnp.mean(xentropy)


def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, learning_rate_fn
):
  dynamic_scale = None
  platform = jax.local_devices()[0].platform
  if config.half_precision and platform == 'gpu':
    dynamic_scale = dynamic_scale_lib.DynamicScale()

  params, batch_stats = initialized(rng, image_size, model)
  tx = optax.sgd(
      learning_rate=learning_rate_fn,
      momentum=config.momentum,
      nesterov=True,
  )
  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      batch_stats=batch_stats,
      dynamic_scale=dynamic_scale,
  )
  return state


def get_fake_batch(batch_size: int = 128) -> dict[str, jnp.ndarray]:
  images = jax.random.uniform(
      jax.random.key(0), (batch_size, 224, 224, 3), dtype=jnp.float32
  )
  labels = jax.random.randint(
      jax.random.key(1), (batch_size,), minval=0, maxval=1000, dtype=jnp.int32
  )
  return {'image': images, 'label': labels}


class BenchmarkResNet(models.ResNet):

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = functools.partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
        axis_name=None,
    )

    x = conv(
        self.num_filters,
        (7, 7),
        (2, 2),
        padding=[(3, 3), (3, 3)],
        name='conv_init',
    )(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(
            self.num_filters * 2**i,
            strides=strides,
            conv=conv,
            norm=norm,
            act=self.act,
        )(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x


def get_apply_fn_and_args(
    config: ml_collections.ConfigDict,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
  if config.model == 'ResNet50':
    model_cls = functools.partial(
        BenchmarkResNet,
        stage_sizes=[3, 4, 6, 3],
        block_cls=models.BottleneckResNetBlock,
    )
  else:
    model_cls = getattr(models, config.model)

  model = create_model(
      model_cls=model_cls, half_precision=config.half_precision
  )

  learning_rate_fn = lambda step: 0.1

  rng = jax.random.key(0)
  image_size = 224
  state = create_train_state(
      rng, config, model, image_size, learning_rate_fn
  )

  batch = get_fake_batch(config.batch_size)

  return (
      bench_train_step,
      (state, batch, learning_rate_fn),
      {},
  )


@functools.partial(jax.jit, static_argnums=(2,))
def bench_train_step(state, batch, learning_rate_fn):

  def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics

  def loss_fn(params):
    logits, new_model_state = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        batch['image'],
        mutable=['batch_stats'],
    )
    loss = cross_entropy_loss(logits, batch['label'])
    weight_penalty_params = jax.tree_util.tree_leaves(params)
    weight_decay = 0.0001
    weight_l2 = sum(jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1)
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_model_state, logits)

  step = state.step
  dynamic_scale = state.dynamic_scale
  lr = learning_rate_fn(step)

  if dynamic_scale:
    grad_fn = dynamic_scale.value_and_grad(loss_fn, has_aux=True)
    dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)

  new_model_state, logits = aux[1]
  metrics = compute_metrics(logits, batch['label'])
  metrics['learning_rate'] = lr

  new_state = state.apply_gradients(
      grads=grads,
      batch_stats=new_model_state['batch_stats'],
  )
  if dynamic_scale:
    new_state = new_state.replace(
        opt_state=jax.tree_util.tree_map(
            functools.partial(jnp.where, is_fin),
            new_state.opt_state,
            state.opt_state,
        ),
        params=jax.tree_util.tree_map(
            functools.partial(jnp.where, is_fin), new_state.params, state.params
        ),
        dynamic_scale=dynamic_scale,
    )
    metrics['scale'] = dynamic_scale.scale

  return new_state, metrics


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_imagenet_trace(state):
  tracing_benchmark.benchmark_tracing(
      get_apply_fn_and_args, imagenet_config.get_config, state
  )


@google_benchmark.register
@google_benchmark.option.unit(google_benchmark.kMillisecond)
def test_flax_imagenet_lower(state):
  tracing_benchmark.benchmark_lowering(
      get_apply_fn_and_args, imagenet_config.get_config, state
  )


if __name__ == '__main__':
  tracing_benchmark.run_benchmarks()
