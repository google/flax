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

from typing import List

import jax
import jax.numpy as jnp
from absl.testing import absltest
from jax import random
import numpy as np

from flax import linen as nn
from flax.core.scope import Array
from flax.linen import summary
from flax import struct
from flax.configurations import use_regular_dict

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

CONSOLE_TEST_KWARGS = dict(force_terminal=False, no_color=True, width=10_000)

def _get_shapes(pytree):
  return jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else x, pytree)

def _get_obj_repr_value(x):
  if isinstance(x, summary._ObjectRepresentation):
    return x.obj
  return x

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

class SummaryTest(absltest.TestCase):

  def test_module_summary(self):
    """
    This test creates a Table using `module_summary` and checks that it
    matches the expected output given the CNN model defined in `_get_tabulate_cnn`.
    """

    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN(test_sow=False)

    table = summary._get_module_table(module, depth=None, show_repeated=True)(
      {"dropout":random.PRNGKey(0), "params": random.PRNGKey(1)},
      x, training=True, mutable=True,
    )
    # get values for inputs and outputs from their _ValueRepresentation
    for row in table:
      row.inputs = jax.tree_util.tree_map(_get_obj_repr_value, row.inputs)
      row.outputs = jax.tree_util.tree_map(_get_obj_repr_value, row.outputs)

    # 10 rows = 1 CNN + 4 ConvBlock_0 + 4 ConvBlock_1 + 1 Dense_0
    self.assertEqual(len(table), 10)

    # check paths
    self.assertEqual(table[0].path, ())

    self.assertEqual(table[1].path, ("block1",))
    self.assertEqual(table[2].path, ("block1", "conv"))
    self.assertEqual(table[3].path, ("block1", "bn"))
    self.assertEqual(table[4].path, ("block1", "dropout"))

    self.assertEqual(table[5].path, ("block2",))
    self.assertEqual(table[6].path, ("block2", "conv"))
    self.assertEqual(table[7].path, ("block2", "bn"))
    self.assertEqual(table[8].path, ("block2", "dropout"))

    self.assertEqual(table[9].path, ("dense",))

    # check outputs shapes
    self.assertEqual(
      (table[0].inputs[0].shape, table[0].inputs[1]),
      (x.shape, dict(training=True)),
    )
    self.assertEqual(
      _get_shapes(table[0].outputs),
      ((batch_size, 10), dict(a=(batch_size, 10), b=(batch_size, 10))),
    )

    self.assertEqual(_get_shapes(table[1].inputs), ((batch_size, 28, 28, 1), {'training': True}))
    self.assertEqual(table[1].outputs.shape, (batch_size, 28, 28, 32))
    self.assertEqual(table[2].inputs.shape, (batch_size, 28, 28, 1))
    self.assertEqual(table[2].outputs.shape, (batch_size, 28, 28, 32))
    self.assertEqual(_get_shapes(table[3].inputs), ((batch_size, 28, 28, 32), {'use_running_average': False}))
    self.assertEqual(table[3].outputs.shape, (batch_size, 28, 28, 32))
    self.assertEqual(_get_shapes(table[4].inputs), ((batch_size, 28, 28, 32), {'deterministic': False}))
    self.assertEqual(table[4].outputs.shape, (batch_size, 28, 28, 32))

    self.assertEqual(_get_shapes(table[5].inputs), ((batch_size, 28, 28, 32), {'training': True}))
    self.assertEqual(table[5].outputs.shape, (batch_size, 28, 28, 64))
    self.assertEqual(table[6].inputs.shape, (batch_size, 28, 28, 32))
    self.assertEqual(table[6].outputs.shape, (batch_size, 28, 28, 64))
    self.assertEqual(_get_shapes(table[7].inputs), ((batch_size, 28, 28, 64), {'use_running_average': False}))
    self.assertEqual(table[7].outputs.shape, (batch_size, 28, 28, 64))
    self.assertEqual(_get_shapes(table[8].inputs), ((batch_size, 28, 28, 64), {'deterministic': False}))
    self.assertEqual(table[8].outputs.shape, (batch_size, 28, 28, 64))

    self.assertEqual(table[9].inputs.shape, (batch_size, 64))
    self.assertEqual(table[9].outputs.shape, (batch_size, 10))

    # check no summary is performed
    for row in table:
      self.assertEqual(
        row.module_variables,
        row.counted_variables,
      )

  @use_regular_dict()
  def test_module_summary_with_depth(self):
    """
    This test creates a Table using `module_summary` set the `depth` argument to `1`,
    table should have less rows as a consequence.
    """
    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN(test_sow=False)

    table = summary._get_module_table(module, depth=1, show_repeated=True)(
      {"dropout":random.PRNGKey(0), "params": random.PRNGKey(1)},
      x, training=True, mutable=True,
    )
    # get values for inputs and outputs from their _ValueRepresentation

    for row in table:
      row.inputs = jax.tree_util.tree_map(_get_obj_repr_value, row.inputs)
      row.outputs = jax.tree_util.tree_map(_get_obj_repr_value, row.outputs)

    # 4 rows = 1 CNN + 1 ConvBlock_0 + 1 ConvBlock_1 + 1 Dense_0
    self.assertEqual(len(table), 4)

    # check paths
    self.assertEqual(table[0].path, ())

    self.assertEqual(table[1].path, ("block1",))
    self.assertEqual(table[2].path, ("block2",))
    self.assertEqual(table[3].path, ("dense",))

    # check outputs shapes
    self.assertEqual(
      (table[0].inputs[0].shape, table[0].inputs[1]),
      (x.shape, dict(training=True)),
    )
    self.assertEqual(
      _get_shapes(table[0].outputs),
      ((batch_size, 10), dict(a=(batch_size, 10), b=(batch_size, 10))),
    )

    self.assertEqual(_get_shapes(table[1].inputs), ((batch_size, 28, 28, 1), {'training': True}))
    self.assertEqual(table[1].outputs.shape, (batch_size, 28, 28, 32))

    self.assertEqual(_get_shapes(table[2].inputs), ((batch_size, 28, 28, 32), {'training': True}))
    self.assertEqual(table[2].outputs.shape, (batch_size, 28, 28, 64))

    self.assertEqual(table[3].inputs.shape, (batch_size, 64))
    self.assertEqual(table[3].outputs.shape, (batch_size, 10))

    # check ConvBlock_0 and ConvBlock_1 are summarized
    self.assertNotEqual(table[1].module_variables, table[1].counted_variables)
    self.assertNotEqual(table[2].module_variables, table[2].counted_variables)

    # check CNN and Dense_0 output are not summarized
    self.assertEqual(table[0].module_variables, table[0].counted_variables)
    self.assertEqual(table[3].module_variables, table[3].counted_variables)


  @use_regular_dict()
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
        console_kwargs=CONSOLE_TEST_KWARGS,
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
    self.assertIn("module", lines[3])
    self.assertIn("inputs", lines[3])
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
      console_kwargs=CONSOLE_TEST_KWARGS,
    )

    self.assertIn("intermediates", module_repr)
    self.assertIn("INTERM", module_repr)

  def test_tabulate_with_method(self):

    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN(test_sow=False)

    module_repr = module.tabulate(
      {"dropout":random.PRNGKey(0), "params": random.PRNGKey(1)},
      x,
      training=True,
      method=CNN.cnn_method,
      console_kwargs=CONSOLE_TEST_KWARGS,
    )

    self.assertIn("(block_method)", module_repr)
    self.assertIn("(cnn_method)", module_repr)

  @use_regular_dict()
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
      console_kwargs=CONSOLE_TEST_KWARGS,
    )(
      x,
      training=True,
    )

    lines = module_repr.split("\n")

    # check title
    module_name = module.__class__.__name__
    self.assertIn(f"{module_name} Summary", lines[1])

    # check headers are correct
    self.assertIn("path", lines[3])
    self.assertIn("module", lines[3])
    self.assertIn("inputs", lines[3])
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


  @use_regular_dict()
  def test_lifted_transform(self):
    class LSTM(nn.Module):
      batch_size: int
      out_feat: int

      @nn.compact
      def __call__(self, x):
          carry = nn.LSTMCell.initialize_carry(
              random.PRNGKey(0), (self.batch_size,), self.out_feat
          )
          Cell = nn.scan(
              nn.LSTMCell,
              variable_broadcast="params",
              split_rngs={"params": False},
              in_axes=1,
              out_axes=1,
          )
          return Cell(name="ScanLSTM")(carry, x)


    lstm = LSTM(batch_size=32, out_feat=128)

    with jax.check_tracer_leaks(True):
      module_repr = lstm.tabulate(
        random.PRNGKey(0),
        x=jnp.ones((32, 128, 64)),
        console_kwargs=CONSOLE_TEST_KWARGS)

    lines = module_repr.splitlines()

    self.assertIn("LSTM", lines[5])
    self.assertIn("ScanLSTM", lines[9])
    self.assertIn("LSTMCell", lines[9])
    self.assertIn("ScanLSTM/ii", lines[13])
    self.assertIn("Dense", lines[13])

  @use_regular_dict()
  def test_lifted_transform_no_rename(self):
    class LSTM(nn.Module):
      batch_size: int
      out_feat: int

      @nn.compact
      def __call__(self, x):
          carry = nn.LSTMCell.initialize_carry(
              random.PRNGKey(0), (self.batch_size,), self.out_feat
          )
          Cell = nn.scan(
              nn.LSTMCell,
              variable_broadcast="params",
              split_rngs={"params": False},
              in_axes=1,
              out_axes=1,
          )
          return Cell()(carry, x)


    lstm = LSTM(batch_size=32, out_feat=128)

    with jax.check_tracer_leaks(True):
      module_repr = lstm.tabulate(
        random.PRNGKey(0),
        x=jnp.ones((32, 128, 64)),
        console_kwargs=CONSOLE_TEST_KWARGS)

    lines = module_repr.splitlines()

    self.assertIn("LSTM", lines[5])
    self.assertIn("ScanLSTMCell_0", lines[9])
    self.assertIn("LSTMCell", lines[9])
    self.assertIn("ScanLSTMCell_0/ii", lines[13])
    self.assertIn("Dense", lines[13])

  @use_regular_dict()
  def test_module_reuse(self):
    class ConvBlock(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = nn.Conv(32, [3, 3])(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.Dropout(0.5, deterministic=True)(x)
        x = nn.relu(x)
        return x

    class CNN(nn.Module):
      @nn.compact
      def __call__(self, x):
        block = ConvBlock()
        x = block(x)
        x = block(x)
        x = block(x)
        return x

    x = jnp.ones((4, 28, 28, 32))
    module_repr = CNN().tabulate(
      jax.random.PRNGKey(0),
      x=x,
      show_repeated=True,
      console_kwargs=CONSOLE_TEST_KWARGS)
    lines = module_repr.splitlines()

    # first call
    self.assertIn("ConvBlock_0/Conv_0", lines[9])
    self.assertIn("bias", lines[9])
    self.assertIn("ConvBlock_0/BatchNorm_0", lines[14])
    self.assertIn("mean", lines[14])
    self.assertIn("bias", lines[14])
    self.assertIn("ConvBlock_0/Dropout_0", lines[19])

    # second call
    self.assertIn("ConvBlock_0/Conv_0", lines[23])
    self.assertNotIn("bias", lines[23])
    self.assertIn("ConvBlock_0/BatchNorm_0", lines[25])
    self.assertNotIn("mean", lines[25])
    self.assertNotIn("bias", lines[25])
    self.assertIn("ConvBlock_0/Dropout_0", lines[27])

    # third call
    self.assertIn("ConvBlock_0/Conv_0", lines[31])
    self.assertNotIn("bias", lines[31])
    self.assertIn("ConvBlock_0/BatchNorm_0", lines[33])
    self.assertNotIn("mean", lines[33])
    self.assertNotIn("bias", lines[33])
    self.assertIn("ConvBlock_0/Dropout_0", lines[35])

  def test_empty_input(self):
    class EmptyInput(nn.Module):
      @nn.compact
      def __call__(self):
        return 1

    module = EmptyInput()
    module_repr = module.tabulate({}, console_kwargs=CONSOLE_TEST_KWARGS)
    lines = module_repr.splitlines()

    self.assertRegex(lines[5], r'|\s*|\s*EmptyInput\s*|\s*|\s*1\s*|')

  def test_numpy_scalar(self):
    class Submodule(nn.Module):
      def __call__(self, x):
        return x + 1

    class EmptyInput(nn.Module):
      @nn.compact
      def __call__(self):
        return Submodule()(x=np.pi)

    module = EmptyInput()
    module_repr = module.tabulate({}, console_kwargs=CONSOLE_TEST_KWARGS)
    lines = module_repr.splitlines()

    self.assertIn('4.141592', lines[5])
    self.assertIn('x: 3.141592', lines[7])
    self.assertIn('4.141592', lines[7])

  @use_regular_dict()
  def test_partitioned_params(self):

    class Classifier(nn.Module):
      @nn.compact
      def __call__(self, x):
        hidden = nn.Dense(
          features=1024,
          kernel_init=nn.with_partitioning(
            nn.initializers.lecun_normal(), (None, 'data')
          ),
          bias_init=nn.with_partitioning(
            nn.initializers.zeros, (None,)
          ),
          name='hidden',
        )
        x = x / 255.0
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.relu(hidden(x))
        x = nn.Dense(features=10, name='head')(x)
        return x

    module = Classifier()
    lines = module.tabulate(jax.random.PRNGKey(0), jnp.empty((1, 28, 28, 1)),
                            console_kwargs=CONSOLE_TEST_KWARGS).splitlines()
    self.assertIn('P(None,)', lines[7])
    self.assertIn('P(None, data)', lines[8])

  def test_non_array_variables(self):

    class Metadata(struct.PyTreeNode):
      names: tuple = struct.field(pytree_node=False)

    class Foo(nn.Module):
      @nn.compact
      def __call__(self):
        self.sow('foo', 'bar', Metadata(('baz', 'qux')))

    module = Foo()
    lines = module.tabulate({},
                            console_kwargs=CONSOLE_TEST_KWARGS).splitlines()
    self.assertIn('names', lines[6])
    self.assertIn('baz', lines[7])
    self.assertIn('qux', lines[8])

  def test_tabulate_param_count(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        h = nn.Dense(4)(x)
        return nn.Dense(2)(h)

    x = jnp.ones((16, 9))
    rep = Foo().tabulate(jax.random.PRNGKey(0), x, console_kwargs=CONSOLE_TEST_KWARGS)
    lines = rep.splitlines()
    self.assertIn('Total Parameters: 50', lines[-2])


if __name__ == '__main__':
  absltest.main()
