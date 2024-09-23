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
import itertools
import numpy as np
import warnings
from functools import partial

from typing import Any
DType = Any

import jax
from jax import custom_jvp, custom_vjp, lax, random
from jax import numpy as jnp
from jax._src import core
from jax._src import dtypes
from jax._src.typing import DTypeLike

try:
  from jax._src import earray
  from jax._src.interpreters import pxla
  CAN_USE_EARRAY = True
except (ModuleNotFoundError, ImportError):
  CAN_USE_EARRAY = False

from flax.linen import initializers, module

OVERWRITE_WITH_GRADIENT = '_overwrite_with_gradient'

# Define a custom dtype for FP8 meta params.
class Fp8MetaTyRules:
  # tell JAX how to lower this dtype to an HLO dtype
  @staticmethod
  def physical_element_aval(dtype) -> core.ShapedArray:
    return core.ShapedArray((), dtype.float_dtype)

  if jax.__version_info__ < (0, 4, 29):
    @staticmethod
    def replicate_trailing_dims(ctx, val, aval):
      del ctx, aval
      return val

    @staticmethod
    def logical_sharding(aval, phys_sharding):
      return phys_sharding

    @staticmethod
    def physical_sharding(aval, sharding):
      return sharding  # unlike KeyTyRules, assume same shape

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

  @staticmethod
  def global_sharded_result_handler(aval, out_sharding, committed):
    if not CAN_USE_EARRAY:
      raise NotImplementedError("convert back under the jit")

    phys_sharding = out_sharding  # unlike KeyTyRules, assume same shape
    phys_aval = core.physical_aval(aval)
    phys_handler_maker = pxla.global_result_handlers[core.ShapedArray]
    phys_handler = phys_handler_maker(phys_aval, phys_sharding, committed)
    return lambda bufs: earray.EArray(aval, phys_handler(bufs))


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
fp32_max_grad = fp8_meta_dtype_wrapper(jnp.float32)

def get_fp8_max(fp8_dtype, out_dtype):
  assert fp8_dtype in (jnp.float8_e4m3fn, jnp.float8_e5m2,
                       jnp.float8_e4m3fnuz, jnp.float8_e5m2fnuz)
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


def quantize_dequantize(x, q_dtype, scale, compute_dtype, quanitze_only=False):
  qx = quantize(x, q_dtype, scale, compute_dtype)
  if quanitze_only:
    return qx
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


def quantize_and_update(
    x, q_dtype, scale, amax_history, compute_dtype, use_direct_quant=False,
    quanitze_only=False,
):
  is_fmax32 = (scale.dtype == fm32 and amax_history.dtype == fm32)
  # convert fm32->f32 so we can do math
  if is_fmax32:
    amax_history = lax.convert_element_type(amax_history, jnp.float32)
    scale = lax.convert_element_type(scale, jnp.float32)

  # Update the fp8 meta
  dtype_max = get_fp8_max(q_dtype, jnp.float32)
  amax_from_history = jnp.max(amax_history, axis=0)

  new_scale = compute_scale(amax_from_history, scale, dtype_max)
  new_history = compute_amax_history(x, amax_history)

  # convert f32->fmax32 so the autodiff system accumulates fp8 meta correctly
  if is_fmax32:
    new_history = lax.convert_element_type(new_history, fp32_max_grad)
    new_scale = lax.convert_element_type(new_scale, fp32_max_grad)

  # Quantize the input
  if not use_direct_quant:
    qx = quantize_dequantize(x, q_dtype, new_scale, compute_dtype, quanitze_only=quanitze_only)
    return qx, new_scale, new_history

  return new_scale, new_history


def dot_general_transpose_lhs(g, x, y, *, dimension_numbers, precision,
                              preferred_element_type: DTypeLike | None,
                              swap_ans=False):
  def _remaining(original, *removed_lists):
    removed = set(itertools.chain(*removed_lists))
    return [i for i in original if i not in removed]

  def _ranges_like(*xs):
    start = 0
    for x in xs:
      x_len = len(x)
      yield range(start, start + x_len)
      start += x_len

  (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
  x_ndim = x.aval.ndim
  x_kept = _remaining(range(x_ndim), x_contract, x_batch)
  y_kept = _remaining(range(np.ndim(y)), y_contract, y_batch)
  if swap_ans:
    ans_batch, ans_y, _ = _ranges_like(x_batch, y_kept, x_kept)
  else:
    ans_batch, _, ans_y = _ranges_like(x_batch, x_kept, y_kept)
  dims = ((ans_y, y_kept), (ans_batch, y_batch))
  x_contract_sorted_by_y = list(np.take(x_contract, np.argsort(y_contract)))
  out_axes = np.argsort(list(x_batch) + x_kept + x_contract_sorted_by_y)
  x_bar = lax.transpose(
    lax.dot_general(
      g, y, dims, precision=precision,
      preferred_element_type=preferred_element_type
    ),
    tuple(out_axes)
  )
  return x_bar

def dot_general_transpose_rhs(g, x, y, *, dimension_numbers, precision,
                              preferred_element_type: DTypeLike | None):
  (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
  swapped_dimension_numbers = ((y_contract, x_contract), (y_batch, x_batch))
  y_bar = dot_general_transpose_lhs(
    g, y, x, dimension_numbers=swapped_dimension_numbers, precision=precision,
    preferred_element_type=preferred_element_type,
    swap_ans=True)
  return y_bar

@partial(custom_vjp, nondiff_argnums=(0, 1))
def in_qdq(compute_dtype, q_dtype, inp, scale, amax_history):
  qin, _, _ = quantize_and_update(
    inp, q_dtype, scale, amax_history, compute_dtype
  )
  return qin


def in_qdq_fwd(compute_dtype, q_dtype, inp, scale, amax_history):
  qin, new_scale, new_history = quantize_and_update(
    inp, q_dtype, scale, amax_history, compute_dtype
  )
  return qin, (new_scale, new_history)


def in_qdq_bwd(compute_dtype, q_dtype, res, g):
  new_scale, new_history = res
  q_g = g
  return q_g, new_scale, new_history


in_qdq.defvjp(in_qdq_fwd, in_qdq_bwd)


@partial(custom_vjp, nondiff_argnums=(0, 1))
def out_qdq(compute_dtype, q_dtype, out, scale, amax_history):
  return out


def out_qdq_fwd(compute_dtype, q_dtype, out, scale, amax_history):
  return out, (scale, amax_history)


def out_qdq_bwd(compute_dtype, q_dtype, res, g):
  scale, amax_history = res
  q_g, new_scale, new_history = quantize_and_update(
    g, q_dtype, scale, amax_history, compute_dtype
  )
  return q_g, new_scale, new_history


out_qdq.defvjp(out_qdq_fwd, out_qdq_bwd)


@partial(custom_vjp, nondiff_argnums=(0, 1, 2))
def in_q(compute_dtype, q_dtype, inp, scale, amax_history):
    qin, new_scale, new_history = quantize_and_update(
        inp, q_dtype, scale, amax_history, compute_dtype, quantize_only=True
    )
    return qin, new_scale

def in_q_fwd(compute_dtype, q_dtype, inp, scale, amax_history):
    qin, new_scale, new_history = quantize_and_update(
        inp, q_dtype, scale, amax_history, compute_dtype, quantize_only=True
    )
    return (qin, new_scale), (new_scale, new_history)

def in_q_bwd(compute_dtype, q_dtype, inp, res, g):
    new_scale, new_history = res
    # We don't compute gradients for inp, scale and amax_history, but we pass through scale and history
    return new_scale, new_history

in_q.defvjp(in_q_fwd, in_q_bwd)

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


def _parse_dot_inputs(*args, **kwargs):
  assert len(args) == 3
  x = args[0]
  k = args[1]
  dimension_numbers = args[2]

  # Use the `k.dtype` since it aligns with the `dtype` of its layers,
  # namely, the computation data type.
  comp_dtype = k.dtype
  x = jnp.asarray(x, comp_dtype)
  return x, k, dimension_numbers, comp_dtype


class Fp8DotGeneralBase(module.Module):
  amax_history_length: int = 1024
  e4m3_dtype: DType = jnp.float8_e4m3fn
  e5m2_dtype: DType = jnp.float8_e5m2

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


class Fp8DotGeneralOp(Fp8DotGeneralBase):
  def __call__(self, *args, **kwargs):
    x, k, dimension_numbers, comp_dtype = _parse_dot_inputs(
      *args, **kwargs
    )
    x_qdq = in_qdq(
      comp_dtype, self.e4m3_dtype, x, self.input_scale.value, self.input_amax_history.value
    )
    k_qdq = in_qdq(
      comp_dtype, self.e4m3_dtype, k, self.kernel_scale.value, self.kernel_amax_history.value
    )

    y_qdq = dot_general_with_precision(x_qdq, k_qdq, dimension_numbers)  # type: ignore
    y = out_qdq(
      comp_dtype,
      self.e5m2_dtype,
      y_qdq,
      self.output_grad_scale.value,
      self.output_grad_amax_history.value,
    )

    return y  # type: ignore

class Fp8DirectDotGeneralOp(Fp8DotGeneralBase):
  def __call__(self, *args, **kwargs):
    x, k, dimension_numbers, comp_dtype = _parse_dot_inputs(
      *args, **kwargs
    )

    q_x, new_input_scale = in_q(
      comp_dtype, jnp.float8_e4m3fn, x, self.input_scale.value, self.input_amax_history)

    y = one_sided_q_dot_dq(
        x,
        q_x,
        new_input_scale, # actualy new lhs scale
        k,
        self.kernel_scale.value,
        self.output_grad_scale.value,
        self.kernel_amax_history.value,
        self.output_grad_amax_history.value,
        comp_dtype,
        dimension_numbers,
        preferred_element_type=x.dtype
    )
    return y

    # y = q_dot_dq(
    #   x,
    #   k,
    #   self.input_scale.value,
    #   self.kernel_scale.value,
    #   self.output_grad_scale.value,
    #   self.input_amax_history.value,
    #   self.kernel_amax_history.value,
    #   self.output_grad_amax_history.value,
    #   comp_dtype,
    #   dimension_numbers,
    #   preferred_element_type=x.dtype
    # )

    # return y  # type: ignore
def one_sided_q_dot_dq_impl(
    lhs,
    q_lhs,
		lhs_scale, # actualy new lhs scale
    rhs,
    rhs_scale,
    out_grad_scale,
    rhs_amax_history,
    out_grad_amax_history,
    compute_dtype,
    dimension_numbers,
    precision,
    preferred_element_type,
    is_training
):
	new_rhs_scale, new_rhs_amax_history = quantize_and_update(
			rhs,
			jnp.float8_e4m3fn,
			rhs_scale,
			rhs_amax_history,
			compute_dtype,
			use_direct_quant=True
	)

	q_rhs = quantize(rhs, jnp.float8_e4m3fn, new_rhs_scale, preferred_element_type)

	out = lax.dot_general(
			q_lhs,
			q_rhs,
			dimension_numbers,
			preferred_element_type=preferred_element_type,
			precision=lax.Precision.DEFAULT,
	)

	out = dequantize(out, preferred_element_type, new_rhs_scale * lhs_scale)
	if is_training:
			res = (
					lhs,
					q_lhs,
					rhs,
					q_rhs,
					new_rhs_scale,
					out_grad_scale,
					new_rhs_amax_history,
					out_grad_amax_history,
			)
			return out, res
	else:
			return out

@partial(custom_vjp, nondiff_argnums=(1,2,8, 9, 10,11))
def one_sided_q_dot_dq(
    lhs,
    q_lhs,
		lhs_scale, # actualy new lhs scale
    rhs,
    rhs_scale,
    out_grad_scale,
    rhs_amax_history,
    out_grad_amax_history,
    compute_dtype,
    dimension_numbers,
    precision=None,
    preferred_element_type=None
):
	return one_sided_q_dot_dq_impl(
			lhs,
			q_lhs,
			lhs_scale,
			rhs,
			rhs_scale,
			out_grad_scale,
			rhs_amax_history,
			out_grad_amax_history,
			compute_dtype,
			dimension_numbers,
			precision,
			preferred_element_type,
			is_training=False,
	)

def one_sided_q_dot_dq_fwd(
    lhs,
    q_lhs,
		lhs_scale,
    rhs,
    rhs_scale,
    out_grad_scale,
    rhs_amax_history,
    out_grad_amax_history,
    compute_dtype,
    dimension_numbers,
    precision,
    preferred_element_type,
):
	return one_sided_q_dot_dq_impl(
			lhs,
			q_lhs,
			rhs,
			rhs_scale,
			out_grad_scale,
			rhs_amax_history,
			out_grad_amax_history,
			compute_dtype,
			dimension_numbers,
			precision,
			preferred_element_type,
			is_training=True
	)

def one_sided_q_dot_dq_bwd(
	  q_lhs,
		lhs_scale,
    compute_dtype,
    dimension_numbers,
    precision,
    preferred_element_type,
    res,
    g
):
	(
		lhs,
		q_lhs,
		rhs,
		q_rhs,
		new_rhs_scale,
		out_grad_scale,
		new_rhs_amax_history,
		out_grad_amax_history,
	) = res

	new_out_grad_scale, new_out_grad_amax_history = quantize_and_update(
			g,
			jnp.float8_e5m2,
			out_grad_scale,
			out_grad_amax_history,
			compute_dtype,
			use_direct_quant=True
	)

	q_g = quantize(g, jnp.float8_e5m2, new_out_grad_scale, preferred_element_type)

	grad_lhs = dot_general_transpose_lhs(
			q_g,
			lhs,
			q_rhs,
			dimension_numbers=dimension_numbers,
			precision=lax.Precision.HIGHEST,
			preferred_element_type=preferred_element_type,
	)
	grad_lhs = dequantize(grad_lhs, preferred_element_type, new_rhs_scale * new_out_grad_scale)

	grad_rhs = dot_general_transpose_rhs(
			q_g,
			q_lhs,
			rhs,
			dimension_numbers=dimension_numbers,
			precision=lax.Precision.HIGHEST,
			preferred_element_type=preferred_element_type,
	)
	grad_rhs = dequantize(grad_rhs, preferred_element_type, lhs_scale * new_out_grad_scale)

	return (
		grad_lhs,
		grad_rhs,
		new_rhs_scale,
		new_out_grad_scale,
		new_rhs_amax_history,
		new_out_grad_amax_history,
	)

one_sided_q_dot_dq.defvjp(one_sided_q_dot_dq_fwd, one_sided_q_dot_dq_bwd)



class NANOOFp8DotGeneralOp(Fp8DotGeneralOp):
  e4m3_dtype: DType = jnp.float8_e4m3fnuz
  e5m2_dtype: DType = jnp.float8_e5m2fnuz
