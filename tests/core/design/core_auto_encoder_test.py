# Copyright 2022 The Flax Authors.
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

from absl.testing import absltest


from flax.core import Scope, Array, init, nn, unfreeze

import jax
from jax import numpy as jnp, random

from flax import struct

from jax.scipy.linalg import expm

from dataclasses import dataclass, InitVar
from typing import Any, Callable, Sequence, NamedTuple, Any


def mlp(scope: Scope, x: Array, hidden: int, out: int):
  x = scope.child(nn.dense, 'hidden')(x, hidden)
  x = nn.relu(x)
  return scope.child(nn.dense, 'out')(x, out)


@dataclass
class AutoEncoder:

  latents: int
  features: int
  hidden: int

  def __call__(self, scope, x):
    z = self.encode(scope, x)
    return self.decode(scope, z)

  def encode(self, scope, x):
    return scope.child(mlp, 'encoder')(x, self.hidden, self.latents)

  def decode(self, scope, z):
    return scope.child(mlp, 'decoder')(z, self.hidden, self.features)


def module_method(fn, name=None):
  if name is None:
    name = fn.__name__ if hasattr(fn, '__name__') else None

  def wrapper(self, *args, **kwargs):
    scope = self.scope.rewound()
    mod_fn = lambda scope: fn(self, scope, *args, **kwargs)
    return scope.child(mod_fn, name)()
  return wrapper


@dataclass
class AutoEncoder2:
  scope: Scope
  latents: int
  features: int
  hidden: int

  def __call__(self, x):
    z = self.encode(x)
    return self.decode(z)

  @module_method
  def encode(self, scope, x):
    return mlp(scope, x, self.hidden, self.latents)

  @module_method
  def decode(self, scope, z):
    return mlp(scope, z, self.hidden, self.features)


@dataclass
class AutoEncoder3:
  encode: Callable
  decode: Callable

  @staticmethod
  def create(scope, hidden: int, latents: int, features: int):
    enc = scope.child(mlp, 'encode', hidden=hidden, out=latents)
    dec = scope.child(mlp, 'decode', hidden=hidden, out=features)
    return AutoEncoder3(enc, dec)

  def __call__(self, x):
    z = self.encode(x)
    return self.decode(z)


class AutoEncoderTest(absltest.TestCase):

  def test_auto_encoder_hp_struct(self):
    ae = AutoEncoder(latents=2, features=4, hidden=3)
    x = jnp.ones((1, 4))
    x_r, variables = init(ae)(random.PRNGKey(0), x)
    self.assertEqual(x.shape, x_r.shape)
    variable_shapes = unfreeze(jax.tree_util.tree_map(jnp.shape, variables['params']))
    self.assertEqual(variable_shapes, {
        'encoder': {
            'hidden': {'kernel': (4, 3), 'bias': (3,)},
            'out': {'kernel': (3, 2), 'bias': (2,)},
        },
        'decoder': {
            'hidden': {'kernel': (2, 3), 'bias': (3,)},
            'out': {'kernel': (3, 4), 'bias': (4,)},
        },
    })

  def test_auto_encoder_with_scope(self):
    ae = lambda scope, x: AutoEncoder2(scope, latents=2, features=4, hidden=3)(x)
    x = jnp.ones((1, 4))

    x_r, variables = init(ae)(random.PRNGKey(0), x)
    self.assertEqual(x.shape, x_r.shape)
    variable_shapes = unfreeze(jax.tree_util.tree_map(jnp.shape, variables['params']))
    self.assertEqual(variable_shapes, {
        'encode': {
            'hidden': {'kernel': (4, 3), 'bias': (3,)},
            'out': {'kernel': (3, 2), 'bias': (2,)},
        },
        'decode': {
            'hidden': {'kernel': (2, 3), 'bias': (3,)},
            'out': {'kernel': (3, 4), 'bias': (4,)},
        },
    })

  def test_auto_encoder_bind_method(self):
    ae = lambda scope, x: AutoEncoder3.create(scope, latents=2, features=4, hidden=3)(x)
    x = jnp.ones((1, 4))

    x_r, variables = init(ae)(random.PRNGKey(0), x)
    self.assertEqual(x.shape, x_r.shape)
    variable_shapes = unfreeze(jax.tree_util.tree_map(jnp.shape, variables['params']))
    self.assertEqual(variable_shapes, {
        'encode': {
            'hidden': {'kernel': (4, 3), 'bias': (3,)},
            'out': {'kernel': (3, 2), 'bias': (2,)},
        },
        'decode': {
            'hidden': {'kernel': (2, 3), 'bias': (3,)},
            'out': {'kernel': (3, 4), 'bias': (4,)},
        },
    })


if __name__ == '__main__':
  absltest.main()
