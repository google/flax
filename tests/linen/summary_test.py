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

import dataclasses
from typing import List, Type

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import lax, random
from jax.nn import initializers

from flax import linen as nn
from flax.core.scope import Array
from flax.linen.summary import _get_module_table

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

def _get_shapes(pytree):
  return jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else x, pytree)

class ConvBlock(nn.Module):
  features: int
  kernel_size: List[int]
  test_sow: bool

  def setup(self) -> None:
    self.conv = nn.Conv(self.features, self.kernel_size)
    self.bn = nn.BatchNorm()
    self.dropout = nn.Dropout(0.5)

  def block_method(self, x: Array, training: bool) -> Array:
    x = self.conv(x)

    if self.test_sow:
      self.sow('intermediates', 'INTERM', x)

    x = self.bn(x, use_running_average=not training)
    x = self.dropout(x, deterministic=not training)
    x = nn.relu(x)
    return x
  
  def __call__(self, x: Array, training: bool) -> Array:
    x = self.conv(x)

    if self.test_sow:
      self.sow('intermediates', 'INTERM', x)

    x = self.bn(x, use_running_average=not training)
    x = self.dropout(x, deterministic=not training)
    x = nn.relu(x)
    return x

class CNN(nn.Module):
  test_sow: bool

  def setup(self) -> None:
    self.block1 = ConvBlock(32, [3, 3], test_sow=self.test_sow)
    self.block2 = ConvBlock(64, [3, 3], test_sow=self.test_sow)
    self.dense = nn.Dense(10)
  
  def cnn_method(self, x: Array, training: bool) -> Array:
    x = self.block1.block_method(x, training=training)
    x = self.block2.block_method(x, training=training)
    x = x.mean(axis=(1, 2))

    if self.test_sow:
      self.sow('intermediates', 'INTERM', x)

    x = self.dense(x)

    return x, dict(a=x, b=x+1.0)

  def __call__(self, x: Array, training: bool) -> Array:
    x = self.block1.block_method(x, training=training)
    x = self.block2.block_method(x, training=training)
    x = x.mean(axis=(1, 2))

    if self.test_sow:
      self.sow('intermediates', 'INTERM', x)

    x = self.dense(x)

    return x, dict(a=x, b=x+1.0)



class ModuleTest(absltest.TestCase):

  def test_module_summary(self):
    """
    This test creates a Table using `module_summary` and checks that it
    matches the expected output given the CNN model defined in `_get_tabulate_cnn`.
    """

    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN(test_sow=False)

    table = _get_module_table(
      module, 
      {"dropout":random.PRNGKey(0), "params": random.PRNGKey(1)},
      method=None, mutable=True, depth=None,
      exclude_methods=set(),
    )( 
      x, training=True
    )

    # 11 rows = 1 Inputs + 4 ConvBlock_0 + 4 ConvBlock_1 + 1 Dense_0 + 1 Module output
    self.assertEqual(len(table), 11)

    # check paths
    self.assertEqual(table[0].path, ("Inputs",))

    self.assertEqual(table[1].path, ("block1", "bn"))
    self.assertEqual(table[2].path, ("block1", "conv"))
    self.assertEqual(table[3].path, ("block1", "dropout"))
    self.assertEqual(table[4].path, ("block1",))

    self.assertEqual(table[5].path, ("block2", "bn"))
    self.assertEqual(table[6].path, ("block2", "conv"))
    self.assertEqual(table[7].path, ("block2", "dropout"))
    self.assertEqual(table[8].path, ("block2",))

    self.assertEqual(table[9].path, ("dense",))
    self.assertEqual(table[10].path, ())

    # check outputs shapes
    self.assertEqual(
      (table[0].outputs[0].shape, table[0].outputs[1]),
      (x.shape, dict(training=True)),
    )

    self.assertEqual(table[1].outputs.shape, (batch_size, 28, 28, 32))
    self.assertEqual(table[2].outputs.shape, (batch_size, 28, 28, 32))
    self.assertEqual(table[3].outputs.shape, (batch_size, 28, 28, 32))
    self.assertEqual(table[4].outputs.shape, (batch_size, 28, 28, 32))

    self.assertEqual(table[5].outputs.shape, (batch_size, 28, 28, 64))
    self.assertEqual(table[6].outputs.shape, (batch_size, 28, 28, 64))
    self.assertEqual(table[7].outputs.shape, (batch_size, 28, 28, 64))
    self.assertEqual(table[8].outputs.shape, (batch_size, 28, 28, 64))

    self.assertEqual(table[9].outputs.shape, (batch_size, 10))
    self.assertEqual(
      _get_shapes(table[10].outputs),
      ((batch_size, 10), dict(a=(batch_size, 10), b=(batch_size, 10))),
    )

    # check no summary is performed
    for row in table:
      self.assertEqual(
        row.module_variables,
        row.counted_variables,
      )
  
  def test_module_summary_with_depth(self):
    """
    This test creates a Table using `module_summary` set the `depth` argument to `1`,
    table should have less rows as a consequence.
    """
    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN(test_sow=False)

    table = _get_module_table(
      module, 
      {"dropout":random.PRNGKey(0), "params": random.PRNGKey(1)},
      method=None, mutable=True, depth=1,
      exclude_methods=set(),
    )(
      x, training=True
    )

    # 5 rows = 1 Inputs + 1 ConvBlock_0 + 1 ConvBlock_1 + 1 Dense_0 + 1 Module output
    self.assertEqual(len(table), 5)

    # check paths
    self.assertEqual(table[0].path, ("Inputs",))
    self.assertEqual(table[1].path, ("block1",))
    self.assertEqual(table[2].path, ("block2",))
    self.assertEqual(table[3].path, ("dense",))
    self.assertEqual(table[4].path, ())

    # check outputs shapes
    self.assertEqual(
      (table[0].outputs[0].shape, table[0].outputs[1]),
      (x.shape, dict(training=True)),
    )
    self.assertEqual(table[1].outputs.shape, (batch_size, 28, 28, 32))
    self.assertEqual(table[2].outputs.shape, (batch_size, 28, 28, 64))
    self.assertEqual(table[3].outputs.shape, (batch_size, 10))
    self.assertEqual(
      _get_shapes(table[4].outputs),
      ((batch_size, 10), dict(a=(batch_size, 10), b=(batch_size, 10))),
    )

    # check ConvBlock_0 and ConvBlock_1 are summarized
    self.assertNotEqual(table[1].module_variables, table[1].counted_variables)
    self.assertNotEqual(table[2].module_variables, table[2].counted_variables)

    # check Dense_0 and Module output are not summarized
    self.assertEqual(table[3].module_variables, table[3].counted_variables)
    self.assertEqual(table[4].module_variables, table[4].counted_variables)

  
  def test_tabulate(self):
    """
    This test creates a string representation of a Module using `Module.tabulate` 
    and checks that it matches the expected output given the CNN model defined in `_get_tabulate_cnn`.
    """
    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN(test_sow=False)

    module_repr = module.tabulate(
        {"dropout":random.PRNGKey(0), "params": random.PRNGKey(1)}, 
        x, 
        training=True,
    )

    # NOTE: its tricky to validate the content of lines
    # because it seems to be shell-dependent, so we will
    # just check lines that wont change between environments
    lines = module_repr.split("\n")

    # check title
    module_name = module.__class__.__name__
    self.assertIn(f"{module_name} Summary", lines[1])

    # check headers are correct
    self.assertIn("path", lines[3])
    self.assertIn("outputs", lines[3])
    self.assertIn("params", lines[3])
    self.assertIn("batch_stats", lines[3])

    # collection counts
    self.assertIn("Total", lines[-6])
    self.assertIn("192", lines[-6])
    self.assertIn("768 B", lines[-6])
    self.assertIn("19,658", lines[-6])
    self.assertIn("78.6 KB", lines[-6])

    # total counts
    self.assertIn("Total Parameters", lines[-3])
    self.assertIn("19,850", lines[-3])
    self.assertIn("79.4 KB", lines[-3])


  def test_tabulate_with_sow(self):

    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN(test_sow=True)

    module_repr = module.tabulate(
      {"dropout":random.PRNGKey(0), "params": random.PRNGKey(1)}, 
      x, 
      training=True,
    )

    self.assertNotIn("INTERM", module_repr)
  
  def test_tabulate_with_method(self):

    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN(test_sow=False)

    module_repr = module.tabulate(
      {"dropout":random.PRNGKey(0), "params": random.PRNGKey(1)}, 
      x, 
      training=True,
      method=CNN.cnn_method,
    )

    self.assertNotIn("INTERM", module_repr)

  def test_tabulate_function(self):
    """
    This test creates a string representation of a Module using `Module.tabulate` 
    and checks that it matches the expected output given the CNN model defined in `_get_tabulate_cnn`.
    """
    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN(test_sow=False)

    module_repr = nn.tabulate(
      module,
      {"dropout":random.PRNGKey(0), "params": random.PRNGKey(1)},
    )(
      x,
      training=True,
    )

    # NOTE: its tricky to validate the content of lines
    # because it seems to be shell-dependent, so we will
    # just check lines that wont change between environments
    lines = module_repr.split("\n")

    # check title
    module_name = module.__class__.__name__
    self.assertIn(f"{module_name} Summary", lines[1])

    # check headers are correct
    self.assertIn("path", lines[3])
    self.assertIn("outputs", lines[3])
    self.assertIn("params", lines[3])
    self.assertIn("batch_stats", lines[3])

    # collection counts
    self.assertIn("Total", lines[-6])
    self.assertIn("192", lines[-6])
    self.assertIn("768 B", lines[-6])
    self.assertIn("19,658", lines[-6])
    self.assertIn("78.6 KB", lines[-6])

    # total counts
    self.assertIn("Total Parameters", lines[-3])
    self.assertIn("19,850", lines[-3])
    self.assertIn("79.4 KB", lines[-3])