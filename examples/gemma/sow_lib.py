# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utilities for sowing intermediate activations."""

import dataclasses
from flax import nnx
import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class LayerIntermediates:
  """Intermediate activations for a single layer."""

  # Dense residual stream activations.
  rs_after_attention: jax.Array | None = None
  rs_after_ffw: jax.Array | None = None

  # Sparse representations for large activations.
  mlp_hidden_topk_values: jax.Array | None = None
  mlp_hidden_topk_indices: jax.Array | None = None
  attn_logits_topk_values: jax.Array | None = None
  attn_logits_topk_indices: jax.Array | None = None

  def merge(self, decoding_step, layer: nnx.Module):
    """Merges the intermediate activations from one step."""

    for field in dataclasses.fields(self.__class__):
      value = getattr(self, field.name)
      if value is None:
        continue
      # We put mlp and attn intermediates into this class without any further
      # nesting. So we have to retrieve the intermediates from the correct
      # sub-module.
      try:
        if field.name.startswith('attn_'):
          step_value = getattr(
              layer.attn, field.name.replace('attn_', '')
          ).value[0]
        elif field.name.startswith('mlp_'):
          step_value = getattr(layer.mlp, field.name.replace('mlp_', '')).value[
              0
          ]
        else:
          step_value = getattr(layer, field.name).value[0]
      except AttributeError as exc:
        raise ValueError(
            f'Intermediate {field.name} is not in the step intermediates.'
        ) from exc
      # This logic is the same for all intermediates. The second dimenions is
      # the length dimension, where we want to merge the intermediates from
      # multiple steps.
      setattr(
          self,
          field.name,
          value.at[:, decoding_step + 1].set(step_value[:, 0, ...]),
      )

  def trim(self, max_length: int):
    """Trims the intermediate activations to the given length."""
    for field in dataclasses.fields(self.__class__):
      value = getattr(self, field.name)
      if value is not None:
        setattr(self, field.name, value[:, :max_length, ...])


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TransformerIntermediates:
  """Intermediate activations of a transformer network."""

  # Embeddings of the input tokens.
  embeddings: jax.Array | None = None

  # Intermediate activations of each layer.
  layers: list[LayerIntermediates] = dataclasses.field(default_factory=list)

  def merge(self, decoding_step, transformer: nnx.Module):
    """Merges the intermediate activations from one step."""
    if self.embeddings is not None:
      try:
        self.embeddings = self.embeddings.at[:, decoding_step + 1, ...].set(
            transformer.embeddings.value[0][:, 0, ...]
        )
      except AttributeError as exc:
        raise ValueError(
            'Embeddings are not in the step intermediates.'
        ) from exc
    if len(self.layers) != len(transformer.layers):
      raise ValueError(
          'Number of layers in the transformer and intermediates do not match.'
      )
    for layer_intermediates, layer_module in zip(
        self.layers, transformer.layers
    ):
      layer_intermediates.merge(decoding_step, layer_module)

  def trim(self, max_length: int):
    """Trims the intermediate activations to the given length."""
    if self.embeddings is not None:
      self.embeddings = self.embeddings[:, :max_length, ...]
    for layer in self.layers:
      layer.trim(max_length)


@dataclasses.dataclass(frozen=True)
class SowConfig:
  """Module for sowing intermediate activations."""

  # Whether to sow embeddings.
  embeddings: bool = False

  # Whether to sow activations after each attention block (in residual stream).
  rs_after_attention: bool = False

  # Whether to sow activations after each FFW block (in residual stream).
  # This is the same as the residual stream activations after a whole layer.
  rs_after_ffw: bool = False

  # If non-zero, top-k activations in a ffw hidden layer are sowed.
  # We use a sparse representation here to save memory.
  mlp_hidden_topk: int = 0

  # If non-zero, top-k attention logits are sowed.
  # We use a sparse representation here to save memory.
  attn_logits_topk: int = 0

  def maybe_sow_embeddings(
      self,
      embeddings: jax.Array,
      module: nnx.Module,
  ):
    """Sows embeddings if configured."""
    if self.embeddings:
      module.sow(nnx.Intermediate, 'embeddings', embeddings)

  def maybe_sow_rs_after_attention(
      self,
      activations: jax.Array,
      module: nnx.Module,
  ):
    """Sows activations after attention if configured."""
    if self.rs_after_attention:
      module.sow(nnx.Intermediate, 'rs_after_attention', activations)

  def maybe_sow_rs_after_ffw(
      self,
      activations: jax.Array,
      module: nnx.Module,
  ):
    """Sows activations after FFW if configured."""
    if self.rs_after_ffw:
      module.sow(nnx.Intermediate, 'rs_after_ffw', activations)

  def maybe_sow_mlp_hidden_topk(
      self,
      activations: jax.Array,
      module: nnx.Module,
  ):
    """Sows top-absolute-k activations in a mlp hidden layer if configured."""
    if self.mlp_hidden_topk:
      _, indices = jax.lax.top_k(jnp.abs(activations), self.mlp_hidden_topk)
      values = jnp.take_along_axis(activations, indices, axis=-1)
      module.sow(nnx.Intermediate, 'hidden_topk_values', values)
      module.sow(nnx.Intermediate, 'hidden_topk_indices', indices)

  def maybe_sow_attn_logits_topk(
      self,
      logits: jax.Array,
      module: nnx.Module,
  ):
    """Sows top-k attention logits if configured."""
    if self.attn_logits_topk:
      values, indices = jax.lax.top_k(logits, self.attn_logits_topk)
      module.sow(nnx.Intermediate, 'logits_topk_values', values)
      module.sow(nnx.Intermediate, 'logits_topk_indices', indices)
