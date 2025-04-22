---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
---

# User Guide on Using FP8

JAX supports various FP8 formats, including E4M3 (jnp.float8_e4m3fn) and E5M2
(jnp.float8_e5m2). Due to the limited range of FP8 data types, higher-precision
data must be scaled to fit within the FP8 representable range, a process known
as quantization (Q). Conversely, de-quantization (DQ) rescales the FP8 data back
to its original type.

While jnp.dot supports FP8 inputs directly, proper quantization and
dequantization is needed for optimal performance. Flax provides
nn.fp8_ops.Fp8DotGeneral and nn.fp8_ops.Fp8Einsum modules that handle
this automatically and can be used with existing layers like nn.Dense.

This tutorial will walk you through the basics of how to use it.

## Setting up our environment

Here, we provide the code necessary to set up the environment for our notebook.
Additionally, we define a function to check if the XLA-optimized HLO will indeed
call an FP8 dot operation under the hood.

*Note: This tutorial relies on the XLA-FP8 feature, which is only supported on
NVIDIA Hopper GPUs or later.*

```{code-cell}
import flax
import jax
import re
import pprint
from jax import random
from jax import numpy as jnp
from jax._src import test_util as jtu
from flax import linen as nn
from flax.linen import fp8_ops

e4m3 = jnp.float8_e4m3fn
f32 = jnp.float32
E4M3_MAX = jnp.finfo(e4m3).max.astype(f32)

assert jtu.is_cuda_compute_capability_at_least("9.0")

def check_fp8_call(lowered):
  hlo = lowered.compile()
  if re.search(r"custom-call\(f8e4m3fn.*, f8e4m3fn.*", hlo.as_text()):
    print("Fp8 call detected!")
  else:
    print("No Fp8 call!")
```

## FLAX Low Level API

The JAX dot operations (e.g. `jnp.dot`) support the FP8 dtype inputs. So it is
legal to do the following call:

```{code-cell}
k0, k1 = random.split(random.key(0), 2)
a = random.uniform(k0, (16, 32))
b = random.uniform(k1, (32, 64))
@jax.jit
def dot_fp8(a, b):
  return jnp.dot(a.astype(e4m3), b.astype(e4m3), preferred_element_type=f32)
check_fp8_call(dot_fp8.lower(a, b))
```

However, this approach has two key limitations:

1. `jnp.dot` does not support custom scaling factors for operands, defaulting to
   a scale of 1.0
2. The autodiff does not automatically use E5M2 for gradients and E4M3 for
   activations/weights during training, which is the recommended practice

To overcome these limitations and implement proper FP8 matrix multiplication, we
recommend using the Flax FP8 APIs. Let's start with a basic scaling approach.


### Current Scaling

Scaling factors are usually defined as `scale = amax(x) / MAX`, where `amax` is
an operation to find the absolute maximum value of the tensor, and `MAX` is the
maximum value of the representable range of the target dtype. This scaling
approach allows us to derive the scaling factors directly from the current
operand tensors of the dot product.

```{code-cell}
@jax.jit
def dot_fp8(a, b):
  a_scale = jnp.max(jnp.abs(A)) / E4M3_MAX
  b_scale = jnp.max(jnp.abs(B)) / E4M3_MAX
  a = fp8_ops.quantize(a, e4m3, a_scale, f32)
  b = fp8_ops.quantize(b, e4m3, b_scale, f32)

  c = jnp.dot(a, b, preferred_element_type=f32)
  c = fp8_ops.dequantize(c, f32, a_scale * b_scale)
  return c

c = dot_fp8(a, b)
check_fp8_call(dot_fp8.lower(a, b))
```

As shown in the code, we perform quantization (`fp8_ops.quantize`) on the
tensors to get the lower precision operands. The `jnp.dot` processes them and
accumulates the output in high precision (i.e., the `preferred_element_type`).
After that, we multiply the result by the scaling factors to dequantize back to
the original range (`fp8_ops.dequantize`). Note that while this example uses
E4M3 for both inputs, it is possible to use different FP8 dtypes like E4M3 and
E5M2 for the inputs. The quantization method and the scaling factors can also be
customized based on application needs.

One major issue with the current scaling method is the performance overhead
introduced by computing `a_scale` and `b_scale`, which requires additional
loading of the operand tensors. To overcome this issue, we recommend the delayed
scaling.

### Delayed Scaling

In delayed scaling, we use a scaling factor associated with an amax history. The
scaling factor remains a scalar, but the amax history is a list that stores amax
values from recent steps (e.g., 1024 steps). Both tensors are computed from
previous steps and maintained in the model parameters.

The quantization and dequantization operations for delayed scaling are provided
by `fp8_ops.in_q` and `fp8_ops.out_dq` respectively. `fp8_ops.in_q` handles
input quantization and update the amax history and scaling factor, while
`fp8_ops.out_dq` performs output dequantization.

```{code-cell}
a_scale = jnp.array(1.0)
b_scale = jnp.array(1.0)
a_amax_hist = jnp.zeros((1024,))
b_amax_hist = jnp.zeros((1024,))

@jax.jit
def dot_fp8(a, a_scale, a_amax_hist, b, b_scale, b_amax_hist):
  a, a_scale = fp8_ops.in_q(f32, e4m3, a, a_scale, a_amax_hist)
  b, b_scale = fp8_ops.in_q(f32, e4m3, b, b_scale, b_amax_hist)
  
  c = jnp.dot(a, b, preferred_element_type=f32)
  c = fp8_ops.out_dq(f32, a_scale, b_scale, c)
  return c

c = dot_fp8(a, a_scale, a_amax_hist, b, b_scale, b_amax_hist)
check_fp8_call(dot_fp8.lower(a, a_scale, a_amax_hist, b, b_scale, b_amax_hist))
```

In this example, we first prepare three pairs of scaling factors and amax
histories, treating them as results computed from previous steps. Then, we apply
`fp8_ops.in_q` to the input operands of `jnp.dot`, followed by `fp8_ops.out_dq`
to the output of `jnp.dot`.


## FLAX High Level API
Flax provides high-level operations to seamlessly integrate FP8 quantization
into existing layers. Instead of manually handling quantization of the delayed
scaling (e.g., the maintanence of the amax history and scaling factors), users
can simply use these drop-in replacements:

* `fp8_ops.Fp8DotGeneral` for `lax.dot_general` operations
* `fp8_ops.Fp8Einsum` for `jnp.einsum` operations 

These operations automatically handle all FP8-related functionality, including
quantization/dequantization, scale factor updates, and FP8 dtype selection for
both forward and backward passes.

Consider the following example:

```{code-cell}
model = nn.Dense(features=64, dot_general_cls=fp8_ops.Fp8DotGeneral)
params = model.init(k0, A)

@jax.jit
def train_step(var, a): 
  c = model.apply(var, a)
  return jnp.sum(c)

check_fp8_call(train_step.lower(params, A))
```

By setting `dot_general_cls=fp8_ops.Fp8DotGeneral`, we replace the
default `lax.dot_general` operation in `nn.Dense` with an FP8-enabled version.
The model usage remains similar, but now includes additional parameters for FP8
quantization: scaling factors and amax history values. The next section explains
how to update these FP8-specific parameters.

For models that use `jnp.einsum` operations, such as Mixture of Experts (MoE)
layers, users can replace them with `fp8_ops.Fp8Einsum` to enable FP8
quantization. Here's an example:

```{code-cell}
from typing import Any
class FooModule(nn.Module):
  einsum: Any = None
  @nn.compact
  def __call__(self, a, b):
    if self.einsum is not None:
      einsum_fn = self.einsum()
    elif self.einsum is None:
      einsum_fn = jnp.einsum
    c = einsum_fn("mk,kn->mn", a, b)
    return c

model = FooModule(einsum=fp8_ops.Fp8Einsum)
params = model.init(k0, a, b)

@jax.jit
def train_step(var, a, b):
  c = model.apply(var, a, b)
  return jnp.sum(c)

check_fp8_call(train_step.lower(params, a, b))
```

## Manipulate FP8 params

The following sections explain the internal FP8 parameters managed by
`fp8_ops.Fp8DotGeneral` and `fp8_ops.Fp8Einsum`. These parameters
include scaling factors and amax history values that control the FP8
quantization process. While most users don't need to interact with these
directly, understanding them can be valuable for advanced optimization and
debugging.

Let's first examine the data structure of `params`. In the code below, we redact
the parameter values and then display the PyTree structure.

```{code-cell}
params_structure = flax.core.unfreeze(params).copy()
params_structure = flax.traverse_util.flatten_dict(params_structure, sep='/')
for key, value in params_structure.items():
    params_structure[key] = '*'
params_structure = flax.traverse_util.unflatten_dict(params_structure, sep='/')
pprint.pprint(params_structure)
```

The output is as follows:

```plaintext
{'_overwrite_with_gradient': {'Fp8Einsum_0': {'input_amax_history': '*',
                                              'input_scale': '*',
                                              'kernel_amax_history': '*',
                                              'kernel_scale': '*',
                                              'output_grad_amax_history': '*',
                                              'output_grad_scale': '*'}}}
```

In addition to the expected `params`, there is an additional category called
`_overwrite_with_gradient`. This category includes three pairs of `amax_history`
and `scale` for the activation, kernel, and dot gradient, respectively.

### Update gradient of FP8 params
Now, we perform one training step to obtain the gradients and see how to use
them to update the parameters.

```{code-cell}
step_fn = jax.jit(jax.grad(train_step, (0, 1)))

grads = step_fn(params, A)

params = flax.core.unfreeze(params)
params = flax.traverse_util.flatten_dict(params, sep='/')
grads = flax.traverse_util.flatten_dict(grads[0], sep='/')

for key, value in params.items():
  if key.startswith('params'):
    params[key] = value + 0.01 * grads[key]
  if key.startswith('_overwrite_with_gradient'):
    params[key] = grads[key]

params = flax.traverse_util.unflatten_dict(params, sep='/')
params = flax.core.freeze(params)
```

The above code demonstrates how to update both `params` and
`_overwrite_with_gradient`. For `params`, we use the formula `new_param =
old_param + 0.01 * grads`, where `0.01` is the learning rate (or users can use
whatever optimizers from `optax`). For `_overwrite_with_gradient`, we simply use
the gradient to overwrite the old values.

Note that `flax.training.train_state.TrainState` conveniently supports the
category of `_overwrite_with_gradient`, so users do not need to modify their
scripts if they don't use custom `TrainState`.

## Accumulate gradient of FP8 params
When the same parameter is used in a branched manner, the autograd mechanism
will add their gradients from these branches. This is common in scenarios like
pipeline parallelism, where each microbatch shares the same set of parameters
for the minibatch. However, for the `_overwrite_with_gradient` parameters, this
accumulation by addition is not meaningful. Instead, we prefer custom
accumulation by taking the maximum value.

To address this, we introduce a custom dtype `fp8_ops.fp32_max_grad`. The basic
usage is demonstrated below:

```{code-cell}
fmax32 = fp8_ops.fp32_max_grad

def reuse_fp8_param(x, y, scale, amax_history):
  scale = scale.astype(fmax32)
  amax_history = amax_history.astype(fmax32)

  x = fp8_ops.in_qdq(f32, e4m3, x, scale, amax_history)
  y = fp8_ops.in_qdq(f32, e4m3, y, scale, amax_history)
  return x + y

reuse_fp8_param_fn = jax.grad(reuse_fp8_param, (0, 1, 2, 3))
reuse_fp8_param_fn = jax.jit(reuse_fp8_param_fn)

_, _, new_ah, new_sf = reuse_fp8_param_fn(2.0, 3.0, a_scale, a_amax_hist)
print(new_ah, new_sf)
```

In this example, we first cast the `scale` and `amax_history` to
`fp8_ops.fp32_max_grad` and then call `fp8_ops.in_qdq` twice using the same pair
of `scale` and `amax_history`. During autograd, their gradients from each branch
will be taken as the maximum, giving us the correct results of:

```plaintext
1.0 [3. 0. 0. ... 0. 0. 0.]
```

If we do not perform the type casting, we get the following result, meaning the
gradients of the two branches are added:

```plaintext
2.0 [5. 0. 0. ... 0. 0. 0.]
```

This casting is already included if users choose to use the high-level APIs.

## Deprecated APIs
Previously, we provided APIs like `fp8_ops.quantize_dequantize` for current
scaling and `fp8_ops.[in|out]_qdq` for delayed scaling. These were used with
high precision dot operations, leveraging an XLA-FP8 feature that
pattern-matched QDQ->dot sequences to Q->fp8_cublas_gemm. The corresponding
high-level API was called `fp8_ops.Fp8DotGeneralOp`. However, this pattern
matching-based solution proved brittle, as the patterns could be easily broken
by other XLA optimizations. We recommend users migrate from these deprecated
APIs to the newer ones described above.

For migration, users should replace:
* `fp8_ops.quantize_dequantize -> jnp.dot` with `fp8_ops.quantize -> jnp.dot ->
  fp8_ops.dequantize`
* `fp8_ops.in_qdq -> jnp.dot -> fp8_ops.out_qdq` with `fp8_ops.in_q -> jnp.dot
  -> fp8_ops.out_dq`
* `fp8_ops.Fp8DotGeneralOp` with `fp8_ops.Fp8DotGeneral`

Additionally, we provide an einsum variant through `fp8_ops.Fp8Einsum`.
