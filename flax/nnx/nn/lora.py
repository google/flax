# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import typing as tp
from types import MappingProxyType

from flax.nnx import rnglib, variablelib
from flax.nnx.module import Module
from flax.nnx.nn import initializers, dtypes
from flax.nnx.nn.linear import Linear
from flax.typing import Dtype, Initializer, PromoteDtypeFn
import jax
import jax.numpy as jnp

Array = jax.Array
Axis = int
Size = int
A = tp.TypeVar('A')

default_a_initializer = initializers.he_uniform()
default_b_initializer = initializers.zeros


class LoRAParam(variablelib.Param[A]):
  pass


class LoRA(Module):
  """A standalone LoRA layer.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp
    >>> layer = nnx.LoRA(3, 2, 4, rngs=nnx.Rngs(0))
    >>> layer.lora_a.shape
    (3, 2)
    >>> layer.lora_b.shape
    (2, 4)
    >>> # Wrap around existing layer
    >>> linear = nnx.Linear(3, 4, rngs=nnx.Rngs(0))
    >>> wrapper = nnx.LoRA(3, 2, 4, base_module=linear, rngs=nnx.Rngs(1))
    >>> assert wrapper.base_module == linear
    >>> wrapper.lora_a.shape
    (3, 2)
    >>> layer.lora_b.shape
    (2, 4)
    >>> y = layer(jnp.ones((16, 3)))
    >>> y.shape
    (16, 4)

  Args:
    in_features: the number of input features.
    lora_rank: the rank of the LoRA dimension.
    out_features: the number of output features.
    base_module: a base module to call and substitute, if possible.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    a_initializer: initializer function for the fan-in matrices. Default to
      `he_uniform`.
    b_initializer: initializer function for the fan-out matrices. Default to
      `zero initializer`.
    lora_param_type: the type of the LoRA params.
    promote_dtype: function to promote the dtype of all input array arguments
      (including Variables accessed through ``self``) to the desired dtype. The
      function should accept a tuple of ``(inputs, lora_a, lora_b)`` and a ``dtype``
      keyword argument, and return a tuple of arrays with the promoted dtype.
    rngs: rng key.
    a_metadata: Optional metadata dictionary to set when initializing
      the fan-in matrices.
    b_metadata: Optional metadata dictionary to set when initializing
      the fan-out matrices.
  """

  def __init__(
    self,
    in_features: int,
    lora_rank: int,
    out_features: int,
    *,
    base_module: tp.Optional[Module] = None,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    a_initializer: Initializer = default_a_initializer,
    b_initializer: Initializer = default_b_initializer,
    lora_param_type: tp.Type[variablelib.Variable] = LoRAParam,
    promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
    rngs: rnglib.Rngs,
    a_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
    b_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
  ):
    self.in_features = in_features
    self.out_features = out_features
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.lora_param_type = lora_param_type
    self.base_module = base_module
    self.promote_dtype = promote_dtype

    self.lora_a = lora_param_type(
      a_initializer(rngs.params(), (in_features, lora_rank), param_dtype),
      **a_metadata,
    )
    self.lora_b = lora_param_type(
      b_initializer(rngs.params(), (lora_rank, out_features), param_dtype),
      **b_metadata,
    )

  def __call__(self, x: jax.Array):
    x, lora_a, lora_b = self.promote_dtype(
      (x, self.lora_a[...], self.lora_b[...]), dtype=self.dtype
    )
    out = x @ lora_a @ lora_b
    if self.base_module is not None:
      if not callable(self.base_module):
        raise ValueError('`self.base_module` must be callable.')
      out += self.base_module(x)
    return out


class LoRALinear(Linear):
  """An `nnx.Linear` layer in which the output will be LoRAified.

  The model state structure will be compatible with that of Linear.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp
    >>> linear = nnx.Linear(3, 4, rngs=nnx.Rngs(0))
    >>> lora_linear = nnx.LoRALinear(3, 4, lora_rank=2, rngs=nnx.Rngs(0))
    >>> linear.kernel.shape
    (3, 4)
    >>> lora_linear.kernel.shape
    (3, 4)
    >>> lora_linear.lora.lora_a.shape
    (3, 2)
    >>> jnp.allclose(linear.kernel[...], lora_linear.kernel[...])
    Array(True, dtype=bool)
    >>> y = lora_linear(jnp.ones((16, 3)))
    >>> y.shape
    (16, 4)

  Args:
    in_features: the number of input features.
    out_features: the number of output features.
    lora_rank: the rank of the LoRA dimension.
    base_module: a base module to call and substitute, if possible.
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    a_initializer: initializer function for the fan-in matrices. Default to
      `he_uniform`.
    b_initializer: initializer function for the fan-out matrices. Default to
      `zero initializer`.
    lora_param_type: the type of the LoRA params.
    lora_promote_dtype: function to promote the dtype for the LoRA submodule.
    a_metadata: Optional metadata dictionary to set when initializing
      the fan-in matrices.
    b_metadata: Optional metadata dictionary to set when initializing
      the fan-out matrices.
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    *,
    lora_rank: int,
    lora_dtype: tp.Optional[Dtype] = None,
    lora_param_dtype: Dtype = jnp.float32,
    a_initializer: Initializer = default_a_initializer,
    b_initializer: Initializer = default_b_initializer,
    lora_param_type: tp.Type[variablelib.Variable] = LoRAParam,
    lora_promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
    rngs: rnglib.Rngs,
    a_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
    b_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
    **kwargs,
  ):
    super().__init__(in_features, out_features, rngs=rngs, **kwargs)
    self.lora = LoRA(
      in_features,
      lora_rank,
      out_features,
      dtype=lora_dtype,
      param_dtype=lora_param_dtype,
      a_initializer=a_initializer,
      b_initializer=b_initializer,
      lora_param_type=lora_param_type,
      promote_dtype=lora_promote_dtype,
      rngs=rngs,
      a_metadata=a_metadata,
      b_metadata=b_metadata,
    )

  def __call__(self, x: jax.Array, out_sharding = None):
    y = super().__call__(x, out_sharding=out_sharding)
    y += self.lora(x)
    return y
