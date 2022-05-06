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

def _maybe_get_shape(x):
  return x.shape if hasattr(x, "shape") else x


def _get_tabulate_cnn(test_sow: bool = False) -> Type[nn.Module]:
  @dataclasses.dataclass
  class ConvBlock(nn.Module):
    features: int
    kernel_size: List[int]
    training: bool

    @nn.compact
    def __call__(self, x: Array) -> Array:
      x = nn.Conv(self.features, self.kernel_size)(x)

      if test_sow:
        self.sow('intermediates', 'INTERM', x)

      x = nn.BatchNorm(use_running_average=not self.training)(x)
      x = nn.Dropout(0.5, deterministic=not self.training)(x)
      x = nn.relu(x)
      return x

  class CNN(nn.Module):
    @nn.compact
    def __call__(self, x: Array, training: bool) -> Array:
      x = ConvBlock(32, [3, 3], training=training)(x)
      x = ConvBlock(64, [3, 3], training=training)(x)
      x = x.mean(axis=(1, 2))

      if test_sow:
        self.sow('intermediates', 'INTERM', x)

      x = nn.Dense(10)(x)

      return x, dict(a=x, b=x+1.0)

  return CNN


class ModuleTest(absltest.TestCase):

  def test_module_summary(self):
    """
    This test creates a Table using `module_summary` and checks that it
    matches the expected output given the CNN model defined in `_get_tabulate_cnn`.
    """

    CNN = _get_tabulate_cnn()
    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN()

    table = _get_module_table(
      module, 
      {"dropout":random.PRNGKey(0), "params": random.PRNGKey(1)},
    )( 
      x, 
      training=True,
    )

    # 11 rows = 1 Inputs + 4 ConvBlock_0 + 4 ConvBlock_1 + 1 Dense_0 + 1 Module output
    self.assertEqual(len(table), 11)

    # check paths
    self.assertEqual(table[0].path, ("Inputs",))

    self.assertEqual(table[1].path, ("ConvBlock_0", "BatchNorm_0"))
    self.assertEqual(table[2].path, ("ConvBlock_0", "Conv_0"))
    self.assertEqual(table[3].path, ("ConvBlock_0", "Dropout_0"))
    self.assertEqual(table[4].path, ("ConvBlock_0",))

    self.assertEqual(table[5].path, ("ConvBlock_1", "BatchNorm_0"))
    self.assertEqual(table[6].path, ("ConvBlock_1", "Conv_0"))
    self.assertEqual(table[7].path, ("ConvBlock_1", "Dropout_0"))
    self.assertEqual(table[8].path, ("ConvBlock_1",))

    self.assertEqual(table[9].path, ("Dense_0",))
    self.assertEqual(table[10].path, ())

    # check outputs shapes
    self.assertEqual(
      jax.tree_map(_maybe_get_shape, table[0].outputs),
      (x.shape, dict(training=True)),
    )

    self.assertEqual(
      jax.tree_map(_maybe_get_shape, table[1].outputs),
      (batch_size, 28, 28, 32),
    )
    self.assertEqual(
      jax.tree_map(_maybe_get_shape, table[2].outputs),
      (batch_size, 28, 28, 32),
    )
    self.assertEqual(
      jax.tree_map(_maybe_get_shape, table[3].outputs),
      (batch_size, 28, 28, 32),
    )
    self.assertEqual(
      jax.tree_map(_maybe_get_shape, table[4].outputs),
      (batch_size, 28, 28, 32),
    )

    self.assertEqual(
      jax.tree_map(_maybe_get_shape, table[5].outputs),
      (batch_size, 28, 28, 64),
    )
    self.assertEqual(
      jax.tree_map(_maybe_get_shape, table[6].outputs),
      (batch_size, 28, 28, 64),
    )
    self.assertEqual(
      jax.tree_map(_maybe_get_shape, table[7].outputs),
      (batch_size, 28, 28, 64),
    )
    self.assertEqual(
      jax.tree_map(_maybe_get_shape, table[8].outputs),
      (batch_size, 28, 28, 64),
    )

    self.assertEqual(
      jax.tree_map(_maybe_get_shape, table[9].outputs),
      (batch_size, 10),
    )
    self.assertEqual(
      jax.tree_map(_maybe_get_shape, table[10].outputs),
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
    CNN = _get_tabulate_cnn()
    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN()

    table = _get_module_table(
      module, 
      {"dropout":random.PRNGKey(0), "params": random.PRNGKey(1)}, 
      depth=1,
    )(
      x, 
      training=True,
    )

    # 5 rows = 1 Inputs + 1 ConvBlock_0 + 1 ConvBlock_1 + 1 Dense_0 + 1 Module output
    self.assertEqual(len(table), 5)

    # check paths
    self.assertEqual(table[0].path, ("Inputs",))
    self.assertEqual(table[1].path, ("ConvBlock_0",))
    self.assertEqual(table[2].path, ("ConvBlock_1",))
    self.assertEqual(table[3].path, ("Dense_0",))
    self.assertEqual(table[4].path, ())

    # check outputs shapes
    self.assertEqual(
      jax.tree_map(_maybe_get_shape, table[0].outputs),
      (x.shape, dict(training=True)),
    )
    self.assertEqual(
      jax.tree_map(_maybe_get_shape, table[1].outputs),
      (batch_size, 28, 28, 32),
    )
    self.assertEqual(
      jax.tree_map(_maybe_get_shape, table[2].outputs),
      (batch_size, 28, 28, 64),
    )
    self.assertEqual(
      jax.tree_map(_maybe_get_shape, table[3].outputs),
      (batch_size, 10),
    )
    self.assertEqual(
      jax.tree_map(_maybe_get_shape, table[4].outputs),
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
    CNN = _get_tabulate_cnn()
    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN()

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

    CNN = _get_tabulate_cnn(test_sow=True)
    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN()

    module_repr = module.tabulate(
      {"dropout":random.PRNGKey(0), "params": random.PRNGKey(1)}, 
      x, 
      training=True,
    )

    self.assertNotIn("INTERM", module_repr)

  def test_tabulate_function(self):
    """
    This test creates a string representation of a Module using `Module.tabulate` 
    and checks that it matches the expected output given the CNN model defined in `_get_tabulate_cnn`.
    """
    CNN = _get_tabulate_cnn()
    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN()

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