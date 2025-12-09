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

from flax.examples.imagenet import models
from flax.examples.imagenet import train
import jax
import jax.numpy as jnp
import ml_collections


def get_fake_batch(batch_size: int = 128) -> dict[str, jnp.ndarray]:
  """Generate a batch of fake ImageNet data.

  Args:
    batch_size: Number of images in the batch.

  Returns:
    A dictionary with 'image' and 'label' keys.
  """
  # ImageNet images: (batch_size, 224, 224, 3)
  images = jax.random.uniform(
      jax.random.key(0), (batch_size, 224, 224, 3), dtype=jnp.float32
  )

  # Labels: integers [0, 1000)
  labels = jax.random.randint(
      jax.random.key(1), (batch_size,), minval=0, maxval=1000, dtype=jnp.int32
  )

  return {'image': images, 'label': labels}


from flax import linen as nn


class BenchmarkResNet(models.ResNet):
  """ResNetV1.5 without axis_name in BatchNorm for single-device benchmarking."""

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = functools.partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
        axis_name=None,  # Changed from 'batch' to None
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
  """Returns the apply function and args for the given config.

  Args:
    config: The training configuration.

  Returns:
    A tuple of the apply function, args, and kwargs.
  """
  # Create model (ResNet50 by default in config)
  # We use BenchmarkResNet to avoid axis_name issues in JIT
  if config.model == 'ResNet50':
    model_cls = functools.partial(
        BenchmarkResNet,
        stage_sizes=[3, 4, 6, 3],
        block_cls=models.BottleneckResNetBlock,
    )
  else:
    # Fallback to original model if not ResNet50 (might fail if it uses axis_name)
    model_cls = getattr(models, config.model)

  model = train.create_model(
      model_cls=model_cls, half_precision=config.half_precision
  )

  # Create learning rate function (needed for train_step)
  # We use a dummy function for benchmarking
  learning_rate_fn = lambda step: 0.1

  # Create train state
  rng = jax.random.key(0)
  image_size = 224
  state = train.create_train_state(
      rng, config, model, image_size, learning_rate_fn
  )

  # Generate fake batch
  batch = get_fake_batch(config.batch_size)

  # Return bench_train_step and its arguments
  return (
      bench_train_step,
      (state, batch, learning_rate_fn),
      {},
  )


@functools.partial(jax.jit, static_argnums=(2,))
def bench_train_step(state, batch, learning_rate_fn):
  """Perform a single training step (JIT-compiled, no pmean)."""

  def compute_metrics(logits, labels):
    loss = train.cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    # metrics = lax.pmean(metrics, axis_name='batch')  # Removed pmean
    return metrics

  def loss_fn(params):
    """loss function used for training."""
    logits, new_model_state = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        batch['image'],
        mutable=['batch_stats'],
    )
    loss = train.cross_entropy_loss(logits, batch['label'])
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
