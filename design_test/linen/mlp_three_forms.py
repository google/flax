import jax
from jax import numpy as jnp, random, lax
from flax import nn
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.linen import Module, MultiModule
import numpy as np

class Dense(Module):
  features: int
  kernel_init: Callable = initializers.lecun_normal()
  bias_init: Callable = initializers.zeros

  def __call__(self, x):
    kernel = self.param('kernel', self.kernel_init, (x.shape[-1], self.features))
    return jnp.dot(x, kernel) + self.param('bias', self.bias_init, (self.features,))

class MLPConcise(Module):
  def __call__(self, x):
    return Dense(self, features=1)(nn.relu(Dense(self, features=2)(x)))

class MLPExplicitWithShapeInference(Module):
  def setup(self):
    self.dense1 = Dense(self, features=2)
    self.dense2 = Dense(self, features=1)

    # Here `self.dense{1,2}.variables` aren't yet materialized --
    # we don't know the input dimensions yet.
    print()
    print("In MLPExplicitWithShapeInference.setup, self.dense2.variables:")
    print(self.dense2.variables)

  def __call__(self, x):
    return self.dense2(nn.relu(self.dense1(x)))

class DenseExplicit(Dense):
  in_features: Optional[int] = None

  def setup(self):
    # Initialize parameters via lazy init in __call__. Assuming we're
    # in a jit, should be "just shape inference".
    self(jnp.zeros((1, self.in_features, )))

class MLPExplicit(Module):
  def setup(self):
    self.dense1 = DenseExplicit(self, in_features=3, features=2)
    self.dense2 = DenseExplicit(self, in_features=2, features=1)
    # explicit instances are materialized immediately at init
    print()
    print("In MLPExplicit.setup, self.dense2.variables:")
    print(self.dense2.variables)

  def __call__(self, x):
    return self.dense2(nn.relu(self.dense1(x)))

rngkey = jax.random.PRNGKey(10)

# For these two, need to init by passing in an input.
mlp_concise = MLPConcise(parent=None).initialized({'param': rngkey}, jnp.zeros((1, 3)))
mlp_explicit_shape_infr = MLPExplicitWithShapeInference(parent=None).initialized({'param': rngkey}, jnp.zeros((1, 3)))

# NOTE: method=None here because we only need to call `setup`
mlp_explicit = MLPExplicit(parent=None).initialized({'param': rngkey}, method=None)

print()
print("mlp_concise vars:")
print(mlp_concise.variables)

print()
print("mlp_explicit_shape_infr vars:")
print(mlp_explicit_shape_infr.variables)

print()
print("mlp_explicit vars:")
print(mlp_explicit.variables)
