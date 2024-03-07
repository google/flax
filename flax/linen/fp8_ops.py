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

import dataclasses
import numpy as np
import warnings
from functools import partial

from jax import custom_jvp, custom_vjp, lax, random
from jax import numpy as jnp
from jax._src import core
from jax._src import dtypes

from flax.linen import initializers, module

OVERWRITE_WITH_GRADIENT = '_overwrite_with_gradient'

# Define a custom dtype for FP8 meta params.
class Fp8MetaTyRules:
  # tell JAX how to lower this dtype to an HLO dtype
  @staticmethod
  def physical_element_aval(dtype) -> core.ShapedArray:
    return core.ShapedArray((), dtype.float_dtype)

  # allow conversions to and from the corresponding float type
  @staticmethod
  def convert_from(fp8_meta_dtype, other_dtype) -> bool:
    return fp8_meta_dtype.float_dtype == other_dtype

  @staticmethod
  def convert_to(other_dtype, fp8_meta_dtype) -> bool:
    return fp8_meta_dtype.float_dtype == other_dtype

  # define how autodiff should accumulate these values
  @staticmethod
  def add(dt, x, y):
    from_fp8_meta = partial(lax.convert_element_type, new_dtype=dt.float_dtype)
    to_fp8_meta = partial(lax.convert_element_type, new_dtype=dt)
    return to_fp8_meta(lax.max(from_fp8_meta(x), from_fp8_meta(y)))

  @staticmethod
  def zero(dt):
    neginf = np.array(-np.inf if dtypes.supports_inf(dt.float_dtype)
                      else dtypes.finfo(dt.float_dtype).min, dt.float_dtype)
    return lax.convert_element_type(neginf, dt)

  @staticmethod
  def tangent_dtype(dtype):
    return dtype

  @staticmethod
  def full(shape, fill_value, dtype):
    fill_value = lax.convert_element_type(fill_value, dtype.float_dtype)
    out_raw = lax.full(shape, fill_value, dtype.float_dtype)
    return lax.convert_element_type(out_raw, dtype)

  # NOTE: by skipping some rules, this dtype can only be used underneath jit
  @staticmethod
  def global_sharded_result_handler(aval, sharding, committed, is_from_xla):
    raise NotImplementedError("convert back under the jit")


# class to use as second argument to jax.dtypes.issubdtype
class fp8_meta_dtype(dtypes.extended): pass

# parameterized datatype for use in e.g. lax.convert_element_type
@dataclasses.dataclass(frozen=True)
class fp8_meta_dtype_wrapper(dtypes.ExtendedDType):
  float_dtype: dtypes.DType
  _rules: type = Fp8MetaTyRules
  type: type = fp8_meta_dtype

  def __repr__(self) -> str:
    nbits = dtypes.finfo(self.float_dtype).bits
    return f'fp8_meta{nbits}'
  name = property(__repr__)

fm32 = fp8_meta_dtype_wrapper(jnp.float32)

def get_fp8_max(fp8_dtype, out_dtype):
  assert fp8_dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2)
  return jnp.finfo(fp8_dtype).max.astype(out_dtype)


def quantize(x, q_dtype, scale, compute_dtype):
  # Explicitly cast the max values to the compute dtype to avoid unnecessary
  # casting to FP32 during the subsequent math operations."
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
  # The algorithm for computing the new scale is sourced from
  #   https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/jax.html#transformer_engine.jax.update_fp8_metas
  # wherein the `original_scale` corresponds to the reciprocal of the `scale`
  # passed in this function.
  scale = 1.0 / scale

  sf = (fp8_max / amax) / (2**margin)
  sf = jnp.where(amax > 0.0, sf, scale)
  sf = jnp.where(jnp.isfinite(amax), sf, scale)

  return 1.0 / sf


def compute_amax_history(x, amax_history):
  amax_update = jnp.max(jnp.abs(x)).astype(amax_history.dtype)
  new_history = jnp.roll(amax_history, shift=-1, axis=0).at[0].set(amax_update)
  return new_history


def qdq_and_return(x, q_dtype, scale, amax_history, compute_dtype):
  is_fm32 = scale.dtype == fm32 and amax_history.dtype == fm32
  # convert fm32->f32 so we can do math
  if is_fm32:
    amax_history = lax.convert_element_type(amax_history, jnp.float32)
    scale = lax.convert_element_type(scale, jnp.float32)

  dtype_max = get_fp8_max(q_dtype, jnp.float32)
  amax_from_history = jnp.max(amax_history, axis=0)
  new_scale = compute_scale(amax_from_history, scale, dtype_max)

  qx = quantize_dequantize(x, q_dtype, new_scale, compute_dtype)

  new_history = compute_amax_history(x, amax_history)

  # convert f32->fm32 so the autodiff system accumulates fp8 meta correctly
  if is_fm32:
    new_history = lax.convert_element_type(new_history, fm32)
    new_scale = lax.convert_element_type(new_scale, fm32)
  return qx, new_scale, new_history


@partial(custom_vjp, nondiff_argnums=(0,))
def in_qdq(compute_dtype, inp, scale, amax_history):
  qin, _, _ = qdq_and_return(
    inp, jnp.float8_e4m3fn, scale, amax_history, compute_dtype
  )
  return qin


def in_qdq_fwd(compute_dtype, inp, scale, amax_history):
  qin, new_scale, new_history = qdq_and_return(
    inp, jnp.float8_e4m3fn, scale, amax_history, compute_dtype
  )
  return qin, (new_scale, new_history)


def in_qdq_bwd(compute_dtype, res, g):
  new_scale, new_history = res
  q_g = g
  return q_g, new_scale, new_history


in_qdq.defvjp(in_qdq_fwd, in_qdq_bwd)


@partial(custom_vjp, nondiff_argnums=(0,))
def out_qdq(compute_dtype, out, scale, amax_history):
  return out


def out_qdq_fwd(compute_dtype, out, scale, amax_history):
  return out, (scale, amax_history)


def out_qdq_bwd(compute_dtype, res, g):
  scale, amax_history = res
  q_g, new_scale, new_history = qdq_and_return(
    g, jnp.float8_e5m2, scale, amax_history, compute_dtype
  )
  return q_g, new_scale, new_history


out_qdq.defvjp(out_qdq_fwd, out_qdq_bwd)


@partial(custom_jvp, nondiff_argnums=(2, 3, 4))
def dot_general_with_precision(
  lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None
):
  if precision != None or preferred_element_type != None:
    warnings.warn(
      'The function dot_general_with_precision will set the '
      'precision/preferred_element_type and disregard any provided '
      'values.'
    )
  return lax.dot_general(
    lhs, rhs, dimension_numbers, precision=lax.Precision.DEFAULT
  )


@dot_general_with_precision.defjvp
def dot_general_with_precision_jvp(
  dimension_numbers, precision, preferred_element_type, primals, tangents
):
  lhs, rhs = primals
  lhs_dot, rhs_dot = tangents

  out = lax.dot_general(
    lhs, rhs, dimension_numbers, precision=lax.Precision.DEFAULT
  )
  grad_out = lax.dot_general(
    lhs_dot, rhs, dimension_numbers, precision=lax.Precision.HIGHEST
  ) + lax.dot_general(
    lhs, rhs_dot, dimension_numbers, precision=lax.Precision.HIGHEST
  )
  return out, grad_out


class Fp8DotGeneralOp(module.Module):
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
      OVERWRITE_WITH_GRADIENT, 'input_amax_history', *amax_history_args
    )
    self.kernel_amax_history = self.variable(
      OVERWRITE_WITH_GRADIENT, 'kernel_amax_history', *amax_history_args
    )
    self.output_grad_amax_history = self.variable(
      OVERWRITE_WITH_GRADIENT, 'output_grad_amax_history', *amax_history_args
    )

    self.input_scale = self.variable(
      OVERWRITE_WITH_GRADIENT, 'input_scale', *scale_args
    )
    self.kernel_scale = self.variable(
      OVERWRITE_WITH_GRADIENT, 'kernel_scale', *scale_args
    )
    self.output_grad_scale = self.variable(
      OVERWRITE_WITH_GRADIENT, 'output_grad_scale', *scale_args
    )

  def __call__(self, *args, **kwargs):
    assert len(args) == 3
    x = args[0]
    k = args[1]
    dimension_numbers = args[2]

    # Use the `k.dtype` since it aligns with the `dtype` of its layers,
    # namely, the computation data type.
    comp_dtype = k.dtype
    x = jnp.asarray(x, comp_dtype)

    x_qdq = in_qdq(
      comp_dtype, x, self.input_scale.value, self.input_amax_history.value
    )
    k_qdq = in_qdq(
      comp_dtype, k, self.kernel_scale.value, self.kernel_amax_history.value
    )
    y_qdq = dot_general_with_precision(x_qdq, k_qdq, dimension_numbers)  # type: ignore
    y = out_qdq(
      comp_dtype,
      y_qdq,
      self.output_grad_scale.value,
      self.output_grad_amax_history.value,
    )

    return y  # type: ignore
