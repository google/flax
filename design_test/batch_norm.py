from flax.core.scope import Scope, init, apply

from flax.nn import initializers

from jax import lax, random, numpy as jnp

def _absolute_dims(rank, dims):
  return tuple([rank + dim if dim < 0 else dim for dim in dims])


def batch_norm(scope: Scope,
               x,
               use_running_average=False,
               axis=-1, momentum=0.99, epsilon=1e-5,
               dtype=jnp.float32,
               bias=True, scale=True,
               bias_init=initializers.zeros, scale_init=initializers.ones,
               axis_name=None, axis_index_groups=None,
               kind='batch_stats'):

  x = jnp.asarray(x, jnp.float32)
  axis = axis if isinstance(axis, tuple) else (axis,)
  axis = _absolute_dims(x.ndim, axis)
  redux = tuple(i for i in range(x.ndim) if i not in axis)

  def pmean(x):
    m = jnp.mean(x, redux, keepdims=True)
    if axis_name is not None:
      m = lax.pmean(m, axis_name=axis_name, axis_index_groups=axis_index_groups)
    return m

  mean = pmean(x)
  squeeze_shape = jnp.squeeze(mean).shape
  mean2 = pmean(jnp.square(x))
  var = mean2 - jnp.square(mean)

  ra_mean = scope.get_variable(kind, 'mean')
  ra_var = scope.get_variable(kind, 'var')

  if use_running_average:
    if ra_mean is not None:
      raise ValueError('batch_stats should be provided if use_running_averages=True')
    mean = jnp.reshape(ra_mean, mean.shape)
    var = jnp.reshape(ra_var, var.shape)
  else:
    if ra_mean is not None:
      beta = 1. - momentum
      ra_mean += beta * (jnp.squeeze(mean) - ra_mean)
      ra_var += beta * (jnp.squeeze(var) - ra_var)
    else:
      ra_mean = jnp.zeros(squeeze_shape)
      ra_var = jnp.ones(squeeze_shape)
    scope.put_variable(kind, 'mean', ra_mean)
    scope.put_variable(kind, 'var', ra_var)

  y = x - mean
  mul = lax.rsqrt(var + epsilon)
  if scale:
    mul = mul * scope.param(
        'scale', scale_init, squeeze_shape).reshape(mean.shape)
  y = y * mul
  if bias:
    y = y + scope.param(
        'bias', bias_init, squeeze_shape).reshape(mean.shape)
  return jnp.asarray(y, dtype)

x = random.normal(random.PRNGKey(0), (2, 3))
y, params = init(batch_norm)(random.PRNGKey(1), x)
print(y)
print(params)
