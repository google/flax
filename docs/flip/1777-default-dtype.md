# FLIP: Default dtypes


- Start Date: 2022-01-11
- FLIP PR: [#1776](https://github.com/google/flax/pull/1776)
- FLIP Issue: [#1777](https://github.com/google/flax/issues/1777)
- Status: Implemented


## Summary

This FLIP proposes to replace the default dtype which is currently fixed to float32, and instead use the JAX type promotion results to derive a default dtype from the input and parameters of a layer.


## Motivation

Currently, Linen Modules always produce `module.dtype` (defaults to float32) outputs regardless of input and parameter dtypes. Half-precision types like float16 and bfloat16 are supported by explicitly passing the half-precision type to each Module. The way this is currently implemented is that each Module has a dtype argument with float32 as the default value. The layer guarantees that this dtype will be the return type of the result returned by `__call__`.

The current behavior is problematic and results in silent bugs, especially for dtypes that do not fit inside float32 (complex, float64). Also, the Linen dtype behavior is significantly different from how NumPy and by extension JAX handle dtypes.


### Dtypes in JAX

JAX uses a NumPy-inspired [dtype promotion](https://github.com/google/jax/blob/main/jax/_src/dtypes.py) mechanism as explained [here](https://jax.readthedocs.io/en/latest/type_promotion.html?highlight=lattice#type-promotion-semantics). The type promotion rules are summarized by the following type lattice:

![JAX type promotion lattice](https://jax.readthedocs.io/en/latest/_images/type_lattice.svg)


## Dtypes in Linen

Besides input arguments, state and in particular parameters could affect dtype promotion. For example: we might feed a float64 input to a Dense layer with float32 parameters. Currently, the result would be truncated to float32. If the input is a complex number the result is even worse because the imaginary part will be silently dropped when casting to float32.

By using the dtype promotion rules already available in JAX we can avoid this issue. A public API is available called `jax.numpy.result_dtype(*args)`, which returns the dtype that JAX would promote the given arguments to, in accordance with the type promotion lattice. For Linen layers the arguments would be the layer inputs together with the parameters. For example, for a linear layer this would be inputs, kernel, and bias.

Note that there is also a `param_dtype` attribute in standard Linen Modules that also defaults to float32. This behavior is left untouched and encodes the common case of having float32 parameters.
There are a few reasons why float32 is almost always the correct dtype for parameters:
1. Storing weights in half-precision often leads to underflow during optimization.
2. Double precision is rarely used because it severely slows down modern accelerators (GPU, TPU). Therefore, such a cost should be explicitly opted-in for.
3. Complex Modules are relatively uncommon. Even within complex networks, the complex inputs can be projected with a real matrix.


# Implementation

A simplified example implementation:


```python
def promote_arrays(*xs, dtype):
 if dtype is None:
   dtype = jnp.result_type(*jax.tree_util.tree_leaves(xs))
 return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype), xs)

Dtype = Any
class Dense(nn.Module):
 features: int
 kernel_init: Callable
 bias_init: Callable
 dtype: Optional[Dtype] = None
 param_dtype: Dtype = jnp.float32

 @nn.compact
 def __call__(self, x):
   kernel = self.param("kernel",
                       self.kernel_init,
                       (x.shape[-1], self.features), self.param_dtype)
   bias = self.param("bias", self.bias_init, (self.features,), self.param_dtype)
   x, kernel, bias = promote_arrays(x, kernel, bias, dtype=self.dtype)
   return x @ kernel + bias
```


## Half-precision dtypes

Some layers don’t work with half-precision dtypes internally. For example: The normalization layers currently compute mean and variance in float32 even when a half-precision dtype is specified to avoid numerical issues. We can replicate this behavior by calling result_dtype with a dummy argument that has the minimum precision for the sub computation to work correctly.


## Backward compatibility

This proposal causes some layers to behave differently in cases where the dtype is not specified to a Linen Module. By default, parameters are in float32. Therefore, passing in half or float32 precision inputs will cause a float32 dtype and no functional differences with current behavior.

When passing complex or float64 precision, the result will no longer truncate the imaginary component or the precision. The silent truncation is problematic and has caused [user complaints](https://github.com/google/flax/issues/805#issuecomment-981468837). Therefore, this change can be considered a bugfix.

Thus, although this proposal strictly speaking changes behavior it is unlikely to cause problems for users. There are 2 exceptions to this which should be rare and easy to fix:
1. A user relies on the enforced float32 to downcast a double precision value.
2. A user relies on the float32 to explicitly upcast a half precision value even though the weights are in half precision.


## Corner cases

In this section we describe corner cases where the implementation of the proposal is not obvious. The two main concerns are how complex numbers are handled in existing layers and how to determine the dtype of state variables.

**Autoregressive decoding cache**

Currently, only attention implements autoregressive caching and the stored key and value mirror the dtype of the key and value passed to the layer. Forcing the cache dtype to be the same as the output dtype could result in reduced precision during cached decoding vs uncached. This seems undesirable. Decision: keep the current behavior.

**Batch statistics**

BatchNorm layers are often used with a half precision output dtype. However, calculating statistics is by default always done in float32 to avoid numerical precision issues and over/underflow for float16. With float64 this would actually cause a downcast so we should now use `np.promote_types(float32, dtype)` such that the precision is at least float32. The running batch statistics will be stored with the same dtype for consistency.

**Complex number support**

Currently, our complex number support is brittle because the default behavior is to truncate the output to the real part. This issue will be fixed by the automatic type promotion proposed in this FLIP. However, some layers require some additional thought to extend to complex numbers correctly:

1. Normalization layers use the complex conjugate to calculate norms instead of normal squaring.
2. Attention: It’s not exactly clear how the dot product and softmax are defined in this case. Raise an error on complex inputs.
3. Recurrent layers: might require special gating / activation functions to function correctly, but these can be specified by the user.


# Discussion

Summarizing the main points from the discussion:


## Consider implicit complex truncation an error

Q:
I'm wondering if we should always raise an error if one of the xs tree leaves is complex but dtype is not. Users should maybe remove imaginary part by themselves if that's really what they want to do.
(Maybe it's a contrived example, but I can imagine cases where layers have their dtype set by parent modules based on assumptions without complex numbers in mind)

A:
This is worth considering in a follow-up CL but this might as well be solved in JAX directly where the safeguard would apply more generally. In NumPy this was also considered but abandoned because it is not backwards compatible.


## Dtype attribute names

Q:
Are the dtype and param_dtype arguments confusing? In particular, should dtype perhaps be called output_dtype to make the difference between the two dtypes more explicit?

A:
This would be a large and orthogonal change wrt to this proposal so leaving it out for now.
Also, this breaks with the standard dtype argument in NumPY/JAX.
Although dtype indeed constrains the output dtype it is also a hint for the dtype we would like the computation to happen in.

