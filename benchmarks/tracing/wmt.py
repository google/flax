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
"""WMT helper functions."""

import functools
from typing import Any

from flax import jax_utils
from flax import linen as nn
from flax.examples.wmt import models
from flax.examples.wmt import train
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
import jax
import jax.numpy as jnp
import ml_collections
import optax


def get_fake_batch(batch_size: int) -> Any:
  """Returns fake data for the given batch size.

  Args:
    batch_size: The global batch size to generate.

  Returns:
    A properly sharded global batch of data.
  """
  rng = jax.random.PRNGKey(0)
  batch = {}
  for k in (
      "inputs",
      "inputs_position",
      "inputs_segmentation",
      "targets",
      "targets_position",
      "targets_segmentation",
  ):
    batch[k] = jax.random.randint(
        rng,
        (batch_size, 256),
        0,
        9999999,
        dtype=jnp.int32,
    )
  batch = common_utils.shard(batch)
  return batch


def get_apply_fn_and_args(
    config: ml_collections.ConfigDict,
    vocab_size: int | None = None,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
  """Returns the apply function and args for the given config.

  Args:
    config: The training configuration.
    vocab_size: The vocabulary size. If None, it will be read from the config.

  Returns:
    A tuple of the apply function, args and kwargs for the apply function, and
    any metadata the training loop needs.
  """
  if vocab_size is None:
    vocab_size = config.vocab_size
  dtype = train.preferred_dtype(config)
  learning_rate_fn = train.create_learning_rate_schedule(
      learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
  )
  train_config = models.TransformerConfig(
      vocab_size=vocab_size,
      output_vocab_size=vocab_size,
      share_embeddings=config.share_embeddings,
      logits_via_embedding=config.logits_via_embedding,
      dtype=dtype,
      emb_dim=config.emb_dim,
      num_heads=config.num_heads,
      num_layers=config.num_layers,
      qkv_dim=config.qkv_dim,
      mlp_dim=config.mlp_dim,
      max_len=max(config.max_target_length, config.max_eval_target_length),
      dropout_rate=config.dropout_rate,
      attention_dropout_rate=config.attention_dropout_rate,
      deterministic=False,
      decode=False,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6),
  )
  p_train_step = jax.pmap(
      functools.partial(
          train.train_step,
          config=train_config,
          learning_rate_fn=learning_rate_fn,
          label_smoothing=config.label_smoothing,
      ),
      axis_name="batch",
      donate_argnums=(0,),
  )  # pytype: disable=wrong-arg-types

  dynamic_scale = None
  if dtype == jnp.float16:
    dynamic_scale = dynamic_scale_lib.DynamicScale()
  eval_config = train_config.replace(deterministic=True)
  m = models.Transformer(eval_config)
  rng = jax.random.key(config.seed)
  rng, init_rng = jax.random.split(rng)
  input_shape = (config.per_device_batch_size, config.max_target_length)
  target_shape = (config.per_device_batch_size, config.max_target_length)
  initial_variables = jax.jit(m.init)(
      init_rng,
      jnp.ones(input_shape, jnp.float32),
      jnp.ones(target_shape, jnp.float32),
  )
  state = train.TrainState.create(
      apply_fn=m.apply,
      params=initial_variables["params"],
      tx=optax.adamw(
          learning_rate=learning_rate_fn,
          b1=0.9,
          b2=0.98,
          eps=1e-9,
          weight_decay=config.weight_decay,
      ),
      dynamic_scale=dynamic_scale,
  )
  state = jax_utils.replicate(state)
  batch = get_fake_batch(
      jax.local_device_count() * config.per_device_batch_size
  )

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap"d training update for performance.
  dropout_rngs = jax.random.split(rng, jax.local_device_count())
  return (
      jax.jit(p_train_step),
      (state, batch),
      dict(dropout_rng=dropout_rngs),
  )
