# Copyright 2023 The Flax Authors.
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

from typing import Callable
from functools import partial

from flax.linen import initializers
from flax.linen.module import Module
from jax import custom_vjp
from jax import lax
from jax import numpy as jnp
from jax import random


# Type annotations
Array = jnp.ndarray
Dtype = jnp.dtype
PRNGKey = jnp.ndarray

class FP8Helper:
  FP8_COLLECTION_NAME: str = "fp8_params"

def get_fp8_max(fp8_dtype, out_dtype):
  assert fp8_dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2)
  return jnp.finfo(fp8_dtype).max.astype(out_dtype)

def quantize(x, q_dtype, scale, compute_dtype):
  # We need to explicitly cast the max value to compute_dtype, otherwise the jax
  # dtype promotion will cast the scaled_x to fp32 in the following ops, which
  # would violate the fp8-matmul pattern matching.
  dtype_max = get_fp8_max(q_dtype, compute_dtype)

  scaled_x = x / jnp.broadcast_to(scale.astype(compute_dtype), x.shape)

  clipped_x = jnp.clip(scaled_x, -dtype_max, dtype_max)

  return clipped_x.astype(q_dtype)

def dequantize(x, dq_dtype, scale):
  return x.astype(dq_dtype) * jnp.broadcast_to(scale.astype(dq_dtype), x.shape)

def quantize_dequantize(x, q_dtype, scale, compute_dtype):
  qx = quantize(x, q_dtype, scale, compute_dtype)
  return dequantize(qx, x.dtype, scale)

def compute_scale(amax, scale, fp8_max, margin=0):
  """Default function to convert amax to scaling factor."""
  # This function copied from the TransformerEngine is used to compute its
  # `scale`. However, our scale matches its `scale_inv` concept. So, we apply
  # the reciprocal operation at the entry and exit of the function.
  scale = 1.0 / scale
  exp = jnp.floor(jnp.log2(fp8_max / amax)) - margin
  sf = jnp.round(lax.pow(2., jnp.abs(exp)))
  sf = jnp.where(amax > 0.0, sf, scale)
  sf = jnp.where(lax.is_finite(amax), sf, scale)
  sf = jnp.where(exp < 0, 1.0 / sf, sf)
  return 1.0 / sf

def compute_scale_and_amax_history(x, q_dtype, scale, amax_history):
  dtype_max = get_fp8_max(q_dtype, jnp.float32)

  amax_update = jnp.max(jnp.abs(x)).astype(scale.dtype)
  new_amax_history = \
      jnp.roll(amax_history, shift=-1, axis=0).at[0].set(amax_update)

  amax_from_history = jnp.max(new_amax_history, axis=0)
  new_scale = compute_scale(amax_from_history, scale, dtype_max)
  return new_scale, new_amax_history

def qdq_and_return(x, q_dtype, scale, amax_history, compute_dtype):
  qx = quantize_dequantize(x, q_dtype, scale, compute_dtype)
  new_scale, new_amax_history = compute_scale_and_amax_history(
      x, q_dtype, scale, amax_history)
  return qx, new_scale, new_amax_history

@partial(custom_vjp, nondiff_argnums=(0,))
def in_qdq(compute_dtype, inp, scale, amax_history):
  qin, _, _ = qdq_and_return(
      inp, jnp.float8_e4m3fn, scale, amax_history, compute_dtype)
  return qin

def in_qdq_fwd(compute_dtype, inp, scale, amax_history):
  qin, new_scale, new_amax_history = qdq_and_return(
      inp, jnp.float8_e4m3fn, scale, amax_history, compute_dtype)
  return qin, (new_scale, new_amax_history)

def in_qdq_bwd(compute_dtype, res, g):
  new_scale, new_amax_history = res
  q_g = g
  return q_g, new_scale, new_amax_history

in_qdq.defvjp(in_qdq_fwd, in_qdq_bwd)


@partial(custom_vjp, nondiff_argnums=(0,))
def out_qdq(compute_dtype, out, scale, amax_history):
  return out

def out_qdq_fwd(compute_dtype, out, scale, amax_history):
  return out, (scale, amax_history)

def out_qdq_bwd(compute_dtype, res, g):
  scale, amax_history = res
  q_g, new_scale, new_amax_history = qdq_and_return(
      g, jnp.float8_e5m2, scale, amax_history, compute_dtype)
  return q_g, new_scale, new_amax_history

out_qdq.defvjp(out_qdq_fwd, out_qdq_bwd)

def fp8_dot_general(lhs, rhs, dimension_numbers, precision, compute_dtype,
                    lhs_scale, lhs_amax_history, rhs_scale, rhs_amax_history,
                    dout_scale, dout_amax_history):
  """Perform dot_general.  """

  lhs_qdq = in_qdq(compute_dtype, lhs, lhs_scale, lhs_amax_history)

  rhs_qdq = in_qdq(compute_dtype, rhs, rhs_scale, rhs_amax_history)

  output_qdq = lax.dot_general(lhs_qdq, rhs_qdq, dimension_numbers, precision)

  out = out_qdq(compute_dtype, output_qdq, dout_scale, dout_amax_history)

  return out


class Fp8DenseGeneralOp(Module):
  amax_history_length: int = 1024

  def setup(self) -> None:
    scale_args = (
        initializers.ones_init(),
        random.PRNGKey(0),
        (1,),
        jnp.float32,
    )
    amax_history_args = (
        initializers.zeros_init(),
        random.PRNGKey(0),
        (self.amax_history_length,),
        jnp.float32,
    )

    self.input_amax_history = self.variable(
        FP8Helper.FP8_COLLECTION_NAME,
        'input_amax_history',
        *amax_history_args)
    self.kernel_amax_history = self.variable(
        FP8Helper.FP8_COLLECTION_NAME,
        'kernel_amax_history',
        *amax_history_args)
    self.output_grad_amax_history = self.variable(
        FP8Helper.FP8_COLLECTION_NAME,
        'output_grad_amax_history',
        *amax_history_args)

    self.input_scale = self.variable(
        FP8Helper.FP8_COLLECTION_NAME,
        'input_scale',
        *scale_args)
    self.kernel_scale = self.variable(
        FP8Helper.FP8_COLLECTION_NAME,
        'kernel_scale',
        *scale_args)
    self.output_grad_scale = self.variable(
        FP8Helper.FP8_COLLECTION_NAME,
        'output_grad_scale',
        *scale_args)


  def __call__(self, *args, **kwargs) -> Array:

    assert len(args) == 3
    inputs = args[0]
    kernel = args[1]
    dimension_numbers = args[2]
    precision = kwargs['precision']
    comp_dtype = kernel.dtype
    inputs = jnp.asarray(inputs, comp_dtype)

    out = fp8_dot_general(inputs, kernel, dimension_numbers, precision,
                          comp_dtype, self.input_scale.value,
                          self.input_amax_history.value,
                          self.kernel_scale.value, self.kernel_amax_history.value,
                          self.output_grad_scale.value,
                          self.output_grad_amax_history.value)
    return out

