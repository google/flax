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
"""Gemma helper functions."""

from typing import Any

from flax import nnx
from flax.examples.gemma import train
from flax.examples.gemma import transformer as transformer_lib
from flax.examples.gemma import utils
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
      'inputs',
      'inputs_position',
      'inputs_segmentation',
      'targets',
      'targets_position',
      'targets_segmentation',
  ):
    batch[k] = jax.random.randint(rng, (batch_size, 128), 0, 9999999, jnp.int32)
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

  # Build Model and Optimizer
  # ---------------------------------------------------------------------------
  if config.transformer_name is not None:
    model_config = transformer_lib.TransformerConfig.from_version_name(
        config.transformer_name,
        num_embed=vocab_size,
        dtype=jnp.bfloat16 if config.use_bfloat16 else jnp.float32,
        axis_rules=config.axis_rules,
    )
  else:
    assert config.transformer_params is not None
    model_config = transformer_lib.TransformerConfig.from_dict(
        **config.transformer_params,
        num_embed=vocab_size,
        dtype=jnp.bfloat16 if config.use_bfloat16 else jnp.float32,
        axis_rules=config.axis_rules,
    )

  # Mesh definition
  devices_array = utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  rng = jax.random.PRNGKey(config.seed)
  rng, init_rng = jax.random.split(rng)

  def constructor(config: transformer_lib.TransformerConfig, key: jax.Array):
    return transformer_lib.Transformer(config, rngs=nnx.Rngs(params=key))

  learning_rate_fn = train.create_learning_rate_schedule(
      learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
  )

  optimizer = optax.adamw(
      learning_rate_fn,
      b1=0.9,
      b2=0.98,
      eps=1e-9,
      weight_decay=config.weight_decay,
  )

  state, state_sharding = utils.setup_initial_state(
      constructor, optimizer, model_config, init_rng, mesh
  )
  data_sharding = jax.NamedSharding(mesh, jax.P(config.data_sharding))
  jit_train_step = jax.jit(
      train.train_step,
      in_shardings=(
          state_sharding,
          data_sharding,
      ),  # type: ignore
      out_shardings=(state_sharding, None),  # type: ignore
      static_argnames=('learning_rate_fn', 'label_smoothing'),
      donate_argnums=0,
  )

  batch = get_fake_batch(config.per_device_batch_size)
  batch = jax.tree.map(lambda x: jnp.asarray(x, device=data_sharding), batch)

  return (
      jit_train_step,
      (state, batch, learning_rate_fn, 0.0),
      dict(),
  )
