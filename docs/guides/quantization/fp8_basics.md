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

Although jnp.dot supports FP8 inputs, certain limitations make it impractical
for real-world applications. Alternatively, XLA, our compiler, can recognize
patterns like <FP8>->DQ->Dot and subsequently invoke FP8 backends (e.g.,
cublasLt for GPUs). FLAX encapsulates such patterns into the
nn.fp8_ops.Fp8DotGeneralOp module, allowing users to easily configure it for
existing layers (e.g., nn.Dense).

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
e5m2 = jnp.float8_e5m2
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
key = random.key(0)
A = random.uniform(key, (16, 32))
B = random.uniform(key, (32, 64))
@jax.jit
def dot_fp8(A, B):
  return jnp.dot(A.astype(e4m3), B.astype(e4m3), preferred_element_type=f32)
check_fp8_call(dot_fp8.lower(A, B))
```

However, there are two main issues with this approach. Firstly, `jnp.dot` does
not accept scaling factors for the operands, defaulting to a scaling factor of
1.0. Secondly, it does not support operands of mixed FP8 data types. For
example, when the operands are E5M2 and E4M3, the dot product is performed using
the promoted FP16 data type.

In real-world scenarios, it is essential to specify scaling factors, either from
calibration for inference or a user-defined algorithm during training.
Additionally, it is common practice to use E5M2 for gradients and E4M3 for
activations and kernels. These limitations make this method less practical for
real-world applications.

To address these limitations and create a more versatile FP8 dot product, we
recommend leveraging XLA-FP8. Let's begin with a simple scaling strategy.


### Current Scaling

Scaling factors are usually defined as `scale = amax(x) / MAX`, where `amax` is
an operation to find the absolute maximum value of the tensor, and `MAX` is the
maximum value of the representable range of the target dtype. This scaling
approach allows us to derive the scaling factors directly from the current
operand tensors of the dot product.

```{code-cell}
@jax.jit
def dot_fp8(A, B):
  A_scale = jnp.max(jnp.abs(A)) / E4M3_MAX
  B_scale = jnp.max(jnp.abs(B)) / E4M3_MAX
  A = fp8_ops.quantize_dequantize(A, e4m3, A_scale, f32)
  B = fp8_ops.quantize_dequantize(B, e4m3, B_scale, f32)

  C = jnp.dot(A, B)
  return C

C = dot_fp8(A, B)
check_fp8_call(dot_fp8.lower(A, B))
```

As shown in the code, we perform fake quantization
(`fp8_ops.quantize_dequantize`) on the operands of the dot product. Although the
`jnp.dot` still processes higher-precision inputs, XLA detects this pattern and
rewrites the dot operation as an FP8 dot call (e.g., cublasLt call for GPUs).
This approach effectively mimics the first example but offers greater
flexibility. We can control the input dtypes (both are set to E4M3 here, but we
could use mixed E4M3 and E5M2) and define scaling factors, which XLA can detect
and use in the dot backend.

One major issue with the current scaling method is the overhead introduced by
computing `A_scale` and `B_scale`, which requires additional loading of the
operand tensors. To overcome this issue, we recommend the delayed scaling.

### Delayed Scaling

In delayed scaling, we use a scaling factor associated with an amax history. The
scaling factor remains a scalar, but the amax history is a list that stores amax
values from recent steps (e.g., 1024 steps). Both tensors are computed from
previous steps and maintained in the model parameters.

Fake quantization for delayed scaling is provided by `fp8_ops.in_qdq` for the
activations and weights, and `fp8_ops.out_qdq` for the gradients.

```{code-cell}
a_scale = jnp.array(1.0)
b_scale = jnp.array(1.0)
g_scale = jnp.array(1.0)
a_amax_hist = jnp.zeros((1024,))
b_amax_hist = jnp.zeros((1024,))
g_amax_hist = jnp.zeros((1024,))

@jax.jit
def dot_fp8(a, a_scale, a_amax_hist, b, b_scale, b_amax_hist,
            g_scale, g_amax_hist):
  a = fp8_ops.in_qdq(f32, e4m3, a, a_scale, a_amax_hist)
  b = fp8_ops.in_qdq(f32, e4m3, b, b_scale, b_amax_hist)
  
  c = jnp.dot(a, b)
  c = fp8_ops.out_qdq(f32, e5m2, c, g_scale, g_amax_hist)
  return c

C = dot_fp8(A, a_scale, a_amax_hist, B, b_scale, b_amax_hist,
            g_scale, g_amax_hist)
check_fp8_call(dot_fp8.lower(A, a_scale, a_amax_hist, B, b_scale, b_amax_hist,
                             g_scale, g_amax_hist))
```

In this example, we first prepare three pairs of scaling factors and amax
histories, treating them as results computed from previous steps. Then, we apply
`fp8_ops.in_qdq` to the input operands of `jnp.dot`, followed by
`fp8_ops.out_qdq` to the output of `jnp.dot`. Note the `fp8_ops.out_qdq` will
apply fake quantization to the gradient of the output via custom_vjp functions.
The new scaling factors and amax histories will be returned through their
gradients, which will be covered in the next section.


## FLAX High Level API
With the FLAX library, incorporating FP8 operations into existing FLAX layers
is a seamless process. Users don't need to manipulate the low-level APIs for
quantization. Instead, they can integrate the provided custom FP8 dot
(`fp8_ops.Fp8DotGeneralOp`) into FLAX layers using a straightforward
"code-injection" approach. This custom operation encapsulates all FP8-related
tasks, including the placement of quantization-dequantization ops, algorithms
for updating scaling factors, and the selection of FP8 dtype combinations for
forward and backward propagation.

Consider the following example:

```{code-cell}
model = nn.Dense(features=64, dot_general_cls=fp8_ops.Fp8DotGeneralOp)
params = model.init(key, A)

@jax.jit
def train_step(var, a): 
  c = model.apply(var, a)
  return jnp.sum(c)

check_fp8_call(train_step.lower(params, A))
```

In this example, we simply set `dot_general_cls=fp8_ops.Fp8DotGeneralOp` to
enable the Dense layer to utilize the FP8 dot operation. The usage of the model
remains almost the same as before. The main difference is the addition of a new
category of parameters: the sets of scaling factors and amax history. In the
next section, we will explore how to update these parameters.

## Manipulate FP8 params
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
{'_overwrite_with_gradient': {'Fp8DotGeneralOp_0': {'input_amax_history': '*',
                                                    'input_scale': '*',
                                                    'kernel_amax_history': '*',
                                                    'kernel_scale': '*',
                                                    'output_grad_amax_history': '*',
                                                    'output_grad_scale': '*'}},
 'params': {'bias': '*', 'kernel': '*'}}
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
