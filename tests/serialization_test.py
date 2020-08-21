# Copyright 2020 The Flax Authors.
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

# Lint as: python3
"""Tests for flax.struct."""

import collections

from typing import Any

from absl.testing import absltest
from flax import nn
from flax import optim
from flax import serialization
from flax import struct

import jax
from jax import random
import jax.numpy as jnp

import numpy as onp

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


@struct.dataclass
class Point:
  x: float
  y: float
  meta: Any = struct.field(pytree_node=False)


class SerializationTest(absltest.TestCase):

  def test_dataclass_serialization(self):
    p = Point(x=1, y=2, meta={'dummy': True})
    state_dict = serialization.to_state_dict(p)
    self.assertEqual(state_dict, {
        'x': 1,
        'y': 2,
    })
    restored_p = serialization.from_state_dict(p, {'x': 3, 'y': 4})
    expected_p = Point(x=3, y=4, meta={'dummy': True})
    self.assertEqual(restored_p, expected_p)

    with self.assertRaises(ValueError):  # invalid field
      serialization.from_state_dict(p, {'z': 3})
    with self.assertRaises(ValueError):  # missing field
      serialization.from_state_dict(p, {'x': 3})

  def test_model_serialization(self):
    rng = random.PRNGKey(0)
    module = nn.Dense.partial(features=1, kernel_init=nn.initializers.ones)
    _, initial_params = module.init_by_shape(rng, [((1, 1), jnp.float32)])
    model = nn.Model(module, initial_params)
    state = serialization.to_state_dict(model)
    self.assertEqual(state, {
        'params': {
            'kernel': onp.ones((1, 1)),
            'bias': onp.zeros((1,)),
        }
    })
    state = {
        'params': {
            'kernel': onp.zeros((1, 1)),
            'bias': onp.zeros((1,)),
        }
    }
    restored_model = serialization.from_state_dict(model, state)
    self.assertEqual(restored_model.params, state['params'])

  def test_optimizer_serialization(self):
    rng = random.PRNGKey(0)
    module = nn.Dense.partial(features=1, kernel_init=nn.initializers.ones)
    _, initial_params = module.init_by_shape(rng, [((1, 1), jnp.float32)])
    model = nn.Model(module, initial_params)
    optim_def = optim.Momentum(learning_rate=1.)
    optimizer = optim_def.create(model)
    state = serialization.to_state_dict(optimizer)
    expected_state = {
        'target': {
            'params': {
                'kernel': onp.ones((1, 1)),
                'bias': onp.zeros((1,)),
            }
        },
        'state': {
            'step': 0,
            'param_states': {
                'params': {
                    'kernel': {'momentum': onp.zeros((1, 1))},
                    'bias': {'momentum': onp.zeros((1,))},
                }
            }
        },
    }
    self.assertEqual(state, expected_state)
    state = jax.tree_map(lambda x: x + 1, expected_state)
    restored_optimizer = serialization.from_state_dict(optimizer, state)
    optimizer_plus1 = jax.tree_map(lambda x: x + 1, optimizer)
    self.assertEqual(restored_optimizer, optimizer_plus1)

  def test_collection_serialization(self):

    @struct.dataclass
    class DummyDataClass:
      x: float

      @classmethod
      def initializer(cls, key, shape):
        del shape, key
        return cls(x=0.)

    class StatefulModule(nn.Module):

      def apply(self):
        state = self.state('state', (), DummyDataClass.initializer)
        state.value = state.value.replace(x=state.value.x + 1.)

    # use stateful
    with nn.stateful() as state:
      self.assertEqual(state.as_dict(), {})
      StatefulModule.init(random.PRNGKey(0))
    self.assertEqual(state.as_dict(), {'/': {'state': DummyDataClass(x=1.)}})
    with nn.stateful(state) as new_state:
      StatefulModule.call({})
    self.assertEqual(new_state.as_dict(),
                     {'/': {
                         'state': DummyDataClass(x=2.)
                     }})
    serialized_state_dict = serialization.to_state_dict(new_state)
    self.assertEqual(serialized_state_dict, {'/': {'state': {'x': 2.}}})
    deserialized_state = serialization.from_state_dict(state,
                                                       serialized_state_dict)
    self.assertEqual(state.as_dict(), {'/': {'state': DummyDataClass(x=1.)}})
    self.assertEqual(new_state.as_dict(), deserialized_state.as_dict())

  def test_numpy_serialization(self):
    normal_dtypes = ['byte', 'b', 'ubyte', 'short',
                     'h', 'ushort', 'i', 'uint', 'intp',
                     'p', 'uintp', 'long', 'l', 'longlong',
                     'q', 'ulonglong', 'half', 'e', 'f',
                     'double', 'd', 'longdouble', 'g',
                     'cfloat', 'cdouble', 'clongdouble', 'm',
                     'bool8', 'b1', 'int64', 'i8', 'uint64', 'u8',
                     'float16', 'f2', 'float32', 'f4', 'float64',
                     'f8', 'float128', 'f16', 'complex64', 'c8',
                     'complex128', 'c16', 'complex256', 'c32',
                     'm8', 'int32', 'i4', 'uint32', 'u4', 'int16',
                     'i2', 'uint16', 'u2', 'int8', 'i1', 'uint8',
                     'u1', 'complex_', 'int0', 'uint0', 'single',
                     'csingle', 'singlecomplex', 'float_', 'intc',
                     'uintc', 'int_', 'longfloat', 'clongfloat',
                     'longcomplex', 'bool_', 'int', 'float',
                     'complex', 'bool']
    onp.random.seed(0)
    for dtype in normal_dtypes:
      for shape in [(), (5,), (10, 10), (1, 20, 30, 1)]:
        arr = onp.random.uniform(-100, 100, size=shape).astype(dtype)
        restored_arr = serialization.msgpack_restore(
            serialization.msgpack_serialize(arr))
        self.assertEqual(restored_arr.dtype, arr.dtype)
        onp.testing.assert_array_equal(restored_arr, arr)

  def test_complex_serialization(self):
    for x in [1j, 1+2j]:
      restored_x = serialization.msgpack_restore(
          serialization.msgpack_serialize(x))
      self.assertEqual(x, restored_x)

  def test_namedtuple_serialization(self):
    foo_class = collections.namedtuple('Foo', 'a b c')
    x1 = foo_class(a=1, b=2, c=3)
    x1_serialized = serialization.to_bytes(x1)
    x2 = foo_class(a=0, b=0, c=0)
    restored_x1 = serialization.from_bytes(x2, x1_serialized)
    self.assertEqual(x1, restored_x1)

  def test_model_serialization_to_bytes(self):
    rng = random.PRNGKey(0)
    module = nn.Dense.partial(features=1, kernel_init=nn.initializers.ones)
    _, initial_params = module.init_by_shape(rng, [((1, 1), jnp.float32)])
    model = nn.Model(module, initial_params)
    serialized_bytes = serialization.to_bytes(model)
    restored_model = serialization.from_bytes(model, serialized_bytes)
    self.assertEqual(restored_model.params, model.params)

  def test_optimizer_serialization_to_bytes(self):
    rng = random.PRNGKey(0)
    module = nn.Dense.partial(features=1, kernel_init=nn.initializers.ones)
    _, initial_params = module.init_by_shape(rng, [((1, 1), jnp.float32)])
    model = nn.Model(module, initial_params)
    optim_def = optim.Momentum(learning_rate=1.)
    optimizer = optim_def.create(model)
    serialized_bytes = serialization.to_bytes(optimizer)
    restored_optimizer = serialization.from_bytes(optimizer, serialized_bytes)
    self.assertEqual(restored_optimizer, optimizer)


if __name__ == '__main__':
  absltest.main()
