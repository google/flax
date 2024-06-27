# Copyright 2024 The Flax Authors.
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

import enum

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import random

from flax import linen as nn
from flax import struct
from flax.core.scope import Array
from flax.linen import summary

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

CONSOLE_TEST_KWARGS = dict(force_terminal=False, no_color=True, width=10_000)


def _get_shapes(pytree):
  return jax.tree_util.tree_map(
    lambda x: x.shape if hasattr(x, 'shape') else x, pytree
  )


def _get_obj_repr_value(x):
  if isinstance(x, summary._ObjectRepresentation):
    return x.obj
  return x


class ConvBlock(nn.Module):
  features: int
  kernel_size: list[int]
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

    return x, dict(a=x, b=x + 1.0)

  def __call__(self, x: Array, training: bool) -> Array:
    x = self.block1.block_method(x, training=training)
    x = self.block2.block_method(x, training=training)
    x = x.mean(axis=(1, 2))

    if self.test_sow:
      self.sow('intermediates', 'INTERM', x)

    x = self.dense(x)

    return x, dict(a=x, b=x + 1.0)


class SummaryTest(absltest.TestCase):
  def test_module_summary(self):
    """
    This test creates a Table using `module_summary` and checks that it
    matches the expected output given the CNN model defined in
    `_get_tabulate_cnn`.
    """

    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN(test_sow=False)

    table = summary._get_module_table(
      module,
      depth=None,
      show_repeated=True,
      compute_flops=True,
      compute_vjp_flops=True,
    )(
      {'dropout': random.key(0), 'params': random.key(1)},
      x,
      training=True,
      mutable=True,
    )
    # get values for inputs and outputs from their _ValueRepresentation
    for row in table:
      row.inputs = jax.tree_util.tree_map(_get_obj_repr_value, row.inputs)
      row.outputs = jax.tree_util.tree_map(_get_obj_repr_value, row.outputs)

    # 10 rows = 1 CNN + 4 ConvBlock_0 + 4 ConvBlock_1 + 1 Dense_0
    self.assertLen(table, 10)

    # check paths
    self.assertEqual(table[0].path, ())

    self.assertEqual(table[1].path, ('block1',))
    self.assertEqual(table[2].path, ('block1', 'conv'))
    self.assertEqual(table[3].path, ('block1', 'bn'))
    self.assertEqual(table[4].path, ('block1', 'dropout'))

    self.assertEqual(table[5].path, ('block2',))
    self.assertEqual(table[6].path, ('block2', 'conv'))
    self.assertEqual(table[7].path, ('block2', 'bn'))
    self.assertEqual(table[8].path, ('block2', 'dropout'))

    self.assertEqual(table[9].path, ('dense',))

    # check outputs shapes
    self.assertEqual(
      (table[0].inputs[0].shape, table[0].inputs[1]),
      (x.shape, dict(training=True)),
    )
    self.assertEqual(
      _get_shapes(table[0].outputs),
      ((batch_size, 10), dict(a=(batch_size, 10), b=(batch_size, 10))),
    )

    self.assertEqual(
      _get_shapes(table[1].inputs),
      ((batch_size, 28, 28, 1), {'training': True}),
    )
    self.assertEqual(table[1].outputs.shape, (batch_size, 28, 28, 32))
    self.assertEqual(table[2].inputs.shape, (batch_size, 28, 28, 1))
    self.assertEqual(table[2].outputs.shape, (batch_size, 28, 28, 32))
    self.assertEqual(
      _get_shapes(table[3].inputs),
      ((batch_size, 28, 28, 32), {'use_running_average': False}),
    )
    self.assertEqual(table[3].outputs.shape, (batch_size, 28, 28, 32))
    self.assertEqual(
      _get_shapes(table[4].inputs),
      ((batch_size, 28, 28, 32), {'deterministic': False}),
    )
    self.assertEqual(table[4].outputs.shape, (batch_size, 28, 28, 32))

    self.assertEqual(
      _get_shapes(table[5].inputs),
      ((batch_size, 28, 28, 32), {'training': True}),
    )
    self.assertEqual(table[5].outputs.shape, (batch_size, 28, 28, 64))
    self.assertEqual(table[6].inputs.shape, (batch_size, 28, 28, 32))
    self.assertEqual(table[6].outputs.shape, (batch_size, 28, 28, 64))
    self.assertEqual(
      _get_shapes(table[7].inputs),
      ((batch_size, 28, 28, 64), {'use_running_average': False}),
    )
    self.assertEqual(table[7].outputs.shape, (batch_size, 28, 28, 64))
    self.assertEqual(
      _get_shapes(table[8].inputs),
      ((batch_size, 28, 28, 64), {'deterministic': False}),
    )
    self.assertEqual(table[8].outputs.shape, (batch_size, 28, 28, 64))

    self.assertEqual(table[9].inputs.shape, (batch_size, 64))
    self.assertEqual(table[9].outputs.shape, (batch_size, 10))

    # check no summary is performed
    for row in table:
      self.assertEqual(
        row.module_variables,
        row.counted_variables,
      )

    # Each module FLOPs >= sum of its submodule FLOPs.
    # Can be greater due to ops like `nn.relu` not belonging to any submodule.
    for r in table:
      flops, vjp_flops = r.flops, r.vjp_flops
      submodule_flops, submodule_vjp_flops = 0, 0
      for s in table:
        if len(s.path) == len(r.path) + 1 and s.path[: len(r.path)] == r.path:
          submodule_flops += s.flops
          submodule_vjp_flops += s.vjp_flops

      self.assertGreaterEqual(flops, submodule_flops)
      self.assertGreaterEqual(vjp_flops, submodule_vjp_flops)

  def test_module_summary_with_depth(self):
    """
    This test creates a Table using `module_summary` set the `depth` argument
    to `1`, table should have fewer rows as a consequence.
    """
    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN(test_sow=False)

    table = summary._get_module_table(
      module,
      depth=1,
      show_repeated=True,
      compute_flops=True,
      compute_vjp_flops=True,
    )(
      {'dropout': random.key(0), 'params': random.key(1)},
      x,
      training=True,
      mutable=True,
    )
    # get values for inputs and outputs from their _ValueRepresentation

    for row in table:
      row.inputs = jax.tree_util.tree_map(_get_obj_repr_value, row.inputs)
      row.outputs = jax.tree_util.tree_map(_get_obj_repr_value, row.outputs)

    # 4 rows = 1 CNN + 1 ConvBlock_0 + 1 ConvBlock_1 + 1 Dense_0
    self.assertLen(table, 4)

    # check paths
    self.assertEqual(table[0].path, ())

    self.assertEqual(table[1].path, ('block1',))
    self.assertEqual(table[2].path, ('block2',))
    self.assertEqual(table[3].path, ('dense',))

    # check outputs shapes
    self.assertEqual(
      (table[0].inputs[0].shape, table[0].inputs[1]),
      (x.shape, dict(training=True)),
    )
    self.assertEqual(
      _get_shapes(table[0].outputs),
      ((batch_size, 10), dict(a=(batch_size, 10), b=(batch_size, 10))),
    )

    self.assertEqual(
      _get_shapes(table[1].inputs),
      ((batch_size, 28, 28, 1), {'training': True}),
    )
    self.assertEqual(table[1].outputs.shape, (batch_size, 28, 28, 32))

    self.assertEqual(
      _get_shapes(table[2].inputs),
      ((batch_size, 28, 28, 32), {'training': True}),
    )
    self.assertEqual(table[2].outputs.shape, (batch_size, 28, 28, 64))

    self.assertEqual(table[3].inputs.shape, (batch_size, 64))
    self.assertEqual(table[3].outputs.shape, (batch_size, 10))

    # check ConvBlock_0 and ConvBlock_1 are summarized
    self.assertNotEqual(table[1].module_variables, table[1].counted_variables)
    self.assertNotEqual(table[2].module_variables, table[2].counted_variables)

    # check CNN and Dense_0 output are not summarized
    self.assertEqual(table[0].module_variables, table[0].counted_variables)
    self.assertEqual(table[3].module_variables, table[3].counted_variables)

    # Top level FLOPs > sum of listed submodule FLOPs, since not all are listed.
    self.assertGreater(table[0].flops, sum(r.flops for r in table[1:]))
    self.assertGreater(table[0].vjp_flops, sum(r.vjp_flops for r in table[1:]))

  def test_tabulate(self):
    """
    This test creates a string representation of a Module using `Module.tabulate`
    and checks that it matches the expected output given the CNN model defined
    in `_get_tabulate_cnn`.
    """
    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN(test_sow=False)

    module_repr = module.tabulate(
      {'dropout': random.key(0), 'params': random.key(1)},
      x,
      training=True,
      console_kwargs=CONSOLE_TEST_KWARGS,
      compute_flops=True,
      compute_vjp_flops=True,
    )

    # NOTE: it's tricky to validate the content of lines
    # because it seems to be shell-dependent, so we will
    # just check lines that won't change between environments
    lines = module_repr.split('\n')

    # check title
    module_name = module.__class__.__name__
    self.assertIn(f'{module_name} Summary', lines[1])

    # check headers are correct
    self.assertIn('path', lines[3])
    self.assertIn('module', lines[3])
    self.assertIn('inputs', lines[3])
    self.assertIn('outputs', lines[3])
    self.assertIn('params', lines[3])
    self.assertIn('flops', lines[3])
    self.assertIn('vjp_flops', lines[3])
    self.assertIn('batch_stats', lines[3])

    # collection counts
    self.assertIn('Total', lines[-6])
    self.assertIn('192', lines[-6])
    self.assertIn('768 B', lines[-6])
    self.assertIn('19,658', lines[-6])
    self.assertIn('78.6 KB', lines[-6])

    # total counts
    self.assertIn('Total Parameters', lines[-3])
    self.assertIn('19,850', lines[-3])
    self.assertIn('79.4 KB', lines[-3])

  def test_tabulate_with_sow(self):
    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN(test_sow=True)

    module_repr = module.tabulate(
      {'dropout': random.key(0), 'params': random.key(1)},
      x,
      training=True,
      mutable=True,
      console_kwargs=CONSOLE_TEST_KWARGS,
      compute_flops=True,
      compute_vjp_flops=True,
    )

    self.assertIn('intermediates', module_repr)
    self.assertIn('INTERM', module_repr)

  def test_tabulate_with_method(self):
    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN(test_sow=False)

    module_repr = module.tabulate(
      {'dropout': random.key(0), 'params': random.key(1)},
      x,
      training=True,
      method=CNN.cnn_method,
      console_kwargs=CONSOLE_TEST_KWARGS,
      compute_flops=True,
      compute_vjp_flops=True,
    )

    self.assertIn('(block_method)', module_repr)
    self.assertIn('(cnn_method)', module_repr)

  def test_tabulate_function(self):
    """
    This test creates a string representation of a Module using
    `Module.tabulate` and checks that it matches the expected output given the
    CNN model defined in `_get_tabulate_cnn`.
    """
    batch_size = 32

    x = jnp.ones((batch_size, 28, 28, 1))
    module = CNN(test_sow=False)

    module_repr = nn.tabulate(
      module,
      {'dropout': random.key(0), 'params': random.key(1)},
      console_kwargs=CONSOLE_TEST_KWARGS,
      compute_flops=True,
      compute_vjp_flops=True,
    )(x, training=True)

    lines = module_repr.split('\n')

    # check title
    module_name = module.__class__.__name__
    self.assertIn(f'{module_name} Summary', lines[1])

    # check headers are correct
    self.assertIn('path', lines[3])
    self.assertIn('module', lines[3])
    self.assertIn('inputs', lines[3])
    self.assertIn('outputs', lines[3])
    self.assertIn('params', lines[3])
    self.assertIn('flops', lines[3])
    self.assertIn('batch_stats', lines[3])

    # collection counts
    self.assertIn('Total', lines[-6])
    self.assertIn('192', lines[-6])
    self.assertIn('768 B', lines[-6])
    self.assertIn('19,658', lines[-6])
    self.assertIn('78.6 KB', lines[-6])

    # total counts
    self.assertIn('Total Parameters', lines[-3])
    self.assertIn('19,850', lines[-3])
    self.assertIn('79.4 KB', lines[-3])

  def test_lifted_transform(self):
    class LSTM(nn.Module):
      features: int

      @nn.compact
      def __call__(self, x):
        carry = nn.LSTMCell(self.features).initialize_carry(
          random.key(0), x[:, 0].shape
        )
        ScanLSTM = nn.scan(
          nn.LSTMCell,
          variable_broadcast='params',
          split_rngs={'params': False},
          in_axes=1,
          out_axes=1,
        )
        return ScanLSTM(self.features, name='ScanLSTM')(carry, x)

    lstm = LSTM(features=128)

    with jax.check_tracer_leaks(True):
      module_repr = lstm.tabulate(
        random.key(0),
        x=jnp.ones((32, 128, 64)),
        console_kwargs=CONSOLE_TEST_KWARGS,
        compute_flops=True,
        compute_vjp_flops=True,
      )

    lines = module_repr.splitlines()

    self.assertIn('LSTM', lines[5])
    self.assertIn('ScanLSTM', lines[9])
    self.assertIn('LSTMCell', lines[9])
    self.assertIn('ScanLSTM/ii', lines[13])
    self.assertIn('Dense', lines[13])

  def test_lifted_transform_no_rename(self):
    class LSTM(nn.Module):
      features: int

      @nn.compact
      def __call__(self, x):
        carry = nn.LSTMCell(self.features).initialize_carry(
          random.key(0), x[:, 0].shape
        )
        ScanLSTM = nn.scan(
          nn.LSTMCell,
          variable_broadcast='params',
          split_rngs={'params': False},
          in_axes=1,
          out_axes=1,
        )
        return ScanLSTM(self.features)(carry, x)

    lstm = LSTM(features=128)

    with jax.check_tracer_leaks(True):
      module_repr = lstm.tabulate(
        random.key(0),
        x=jnp.ones((32, 128, 64)),
        console_kwargs=CONSOLE_TEST_KWARGS,
        compute_flops=True,
        compute_vjp_flops=True,
      )

    lines = module_repr.splitlines()

    self.assertIn('LSTM', lines[5])
    self.assertIn('ScanLSTMCell_0', lines[9])
    self.assertIn('LSTMCell', lines[9])
    self.assertIn('ScanLSTMCell_0/ii', lines[13])
    self.assertIn('Dense', lines[13])

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
      jax.random.key(0),
      x=x,
      show_repeated=True,
      console_kwargs=CONSOLE_TEST_KWARGS,
      compute_flops=True,
      compute_vjp_flops=True,
    )
    lines = module_repr.splitlines()

    # first call
    self.assertIn('ConvBlock_0/Conv_0', lines[9])
    self.assertIn('bias', lines[9])
    self.assertIn('ConvBlock_0/BatchNorm_0', lines[14])
    self.assertIn('mean', lines[14])
    self.assertIn('bias', lines[14])
    self.assertIn('ConvBlock_0/Dropout_0', lines[19])

    # second call
    self.assertIn('ConvBlock_0/Conv_0', lines[23])
    self.assertNotIn('bias', lines[23])
    self.assertIn('ConvBlock_0/BatchNorm_0', lines[25])
    self.assertNotIn('mean', lines[25])
    self.assertNotIn('bias', lines[25])
    self.assertIn('ConvBlock_0/Dropout_0', lines[27])

    # third call
    self.assertIn('ConvBlock_0/Conv_0', lines[31])
    self.assertNotIn('bias', lines[31])
    self.assertIn('ConvBlock_0/BatchNorm_0', lines[33])
    self.assertNotIn('mean', lines[33])
    self.assertNotIn('bias', lines[33])
    self.assertIn('ConvBlock_0/Dropout_0', lines[35])

    # Test that CNN FLOPs are 3x ConvBlock FLOPs.
    args = ({'dropout': random.key(0), 'params': random.key(1)}, x)
    cnn = summary._get_module_table(
      CNN(),
      depth=1,
      show_repeated=True,
      compute_flops=True,
      compute_vjp_flops=True,
    )(*args, mutable=True)

    block = summary._get_module_table(
      ConvBlock(),
      depth=1,
      show_repeated=True,
      compute_flops=True,
      compute_vjp_flops=True,
    )(*args, mutable=True)

    # Total forward/backward FLOPs equal to their sums of sub blocks.
    self.assertEqual(cnn[0].flops, sum(r.flops for r in cnn[1:]))
    self.assertEqual(cnn[0].vjp_flops, sum(r.vjp_flops for r in cnn[1:]))

    # Each sub block has cost equal to ConvBlock instantiated separately.
    for r in cnn[1:]:
      self.assertEqual(r.flops, block[0].flops)
      self.assertEqual(r.vjp_flops, block[0].vjp_flops)

  def test_empty_input(self):
    class EmptyInput(nn.Module):
      @nn.compact
      def __call__(self):
        return 1

    module = EmptyInput()
    module_repr = module.tabulate(
      {},
      console_kwargs=CONSOLE_TEST_KWARGS,
      compute_flops=True,
      compute_vjp_flops=True,
    )
    lines = module_repr.splitlines()

    # 1 output and 0 forward / backward FLOPs.
    self.assertRegex(
      lines[5], r'│\s*│\s*EmptyInput\s*│\s*│\s*1\s*│\s*0\s*│\s*0\s*│'
    )

  def test_numpy_scalar(self):
    class Submodule(nn.Module):
      def __call__(self, x):
        return x + 1

    class EmptyInput(nn.Module):
      @nn.compact
      def __call__(self):
        return Submodule()(x=np.pi)

    module = EmptyInput()
    module_repr = module.tabulate(
      {},
      console_kwargs=CONSOLE_TEST_KWARGS,
      compute_flops=True,
      compute_vjp_flops=True,
    )
    lines = module_repr.splitlines()

    self.assertIn('4.141592', lines[5])
    self.assertIn('x: 3.141592', lines[7])
    self.assertIn('4.141592', lines[7])

    # 0 forward / backward FLOPs due to precomputed output values.
    self.assertIn('│ 0     │ 0', lines[5])
    self.assertIn('│ 0     │ 0', lines[7])

  def test_partitioned_params(self):
    class Classifier(nn.Module):
      @nn.compact
      def __call__(self, x):
        hidden = nn.Dense(
          features=1024,
          kernel_init=nn.with_partitioning(
            nn.initializers.lecun_normal(), (None, 'data')
          ),
          bias_init=nn.with_partitioning(nn.initializers.zeros, (None,)),
          name='hidden',
        )
        x = x / 255.0
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.relu(hidden(x))
        x = nn.Dense(features=10, name='head')(x)
        return x

    module = Classifier()
    lines = module.tabulate(
      jax.random.key(0),
      jnp.empty((1, 28, 28, 1)),
      console_kwargs=CONSOLE_TEST_KWARGS,
      compute_flops=True,
      compute_vjp_flops=True,
    ).splitlines()
    self.assertIn('P(None,)', lines[7])
    self.assertIn('P(None, data)', lines[8])

    # Per-layer forward FLOPs:
    self.assertIn('1606656', lines[7])  # 1 * (28 * 28 * 1) * 1024 * 2 + 1024
    self.assertIn('20490', lines[12])  #  1 * (1024       ) * 10   * 2 + 10

    # Total forward FLOPs: input division + ReLU + two dense layers above.
    # (1 * 28 * 28 * 1) + (1 * 1024) + 1606656 + 20490.
    self.assertIn('1628954', lines[5])

    # Per-layer backward FLOPs:

    # [3x MMs: forward, input cotangent, weight cotangent] 1024 * 784 * 2 * 3
    # + [forward bias addition] 1024
    # + [`mutable=True`: weight and bias sizes] 1024 * 784 + 1024
    self.assertIn('5621760', lines[7])

    # [3x matmuls: forward, input cotangent, weight cotangent] 1024 * 10 * 2 * 3
    # + [forward bias addition] 10
    # + [`mutable=True`: weight and bias sizes] 1024 * 10 + 10
    self.assertIn('71700', lines[12])

    # Total backward FLOPs: input division + ReLU + two dense layers above.
    # 2 * (1 * 28 * 28 * 1) + 3 * (1 * 1024) + 5621760 + 71700.
    self.assertIn('5698100', lines[5])

  def test_non_array_variables(self):
    class Metadata(struct.PyTreeNode):
      names: tuple = struct.field(pytree_node=False)

    class Foo(nn.Module):
      @nn.compact
      def __call__(self):
        self.sow('foo', 'bar', Metadata(('baz', 'qux')))

    module = Foo()
    lines = module.tabulate(
      {},
      console_kwargs=CONSOLE_TEST_KWARGS,
      compute_flops=True,
      compute_vjp_flops=True,
    ).splitlines()
    self.assertIn('names', lines[6])
    self.assertIn('baz', lines[7])
    self.assertIn('qux', lines[8])

    # 0 forward and backward FLOPs.
    self.assertIn('│ 0     │ 0', lines[5])

  def test_tabulate_param_count_and_flops(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        h = nn.Dense(4)(x)
        return nn.Dense(2)(h)

    module = Foo()
    rng = jax.random.key(0)
    x = jnp.ones((16, 9))

    rep = module.tabulate(
      rng,
      x,
      console_kwargs=CONSOLE_TEST_KWARGS,
      compute_flops=True,
      compute_vjp_flops=True,
    )
    lines = rep.splitlines()
    self.assertIn('Total Parameters: 50', lines[-2])

  def test_tabulate_enum(self):
    class Net(nn.Module):
      @nn.compact
      def __call__(self, inputs):
        x = inputs['x']
        x = nn.Dense(features=2)(x)
        return jnp.sum(x)

    class InputEnum(str, enum.Enum):
      x = 'x'

    inputs = {InputEnum.x: jnp.ones((1, 1))}
    # test args
    lines = Net().tabulate(jax.random.key(0), inputs).split('\n')
    self.assertIn('x: \x1b[2mfloat32\x1b[0m[1,1]', lines[5])
    # test kwargs
    lines = Net().tabulate(jax.random.key(0), inputs=inputs).split('\n')
    self.assertIn('inputs:', lines[5])
    self.assertIn('x: \x1b[2mfloat32\x1b[0m[1,1]', lines[6])

  def test_tabulate_norm_wrapper(self):
    class SubModel(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = nn.SpectralNorm(nn.Dense(5))(x, update_stats=False)
        x = nn.Dense(6)(x)
        x = nn.WeightNorm(nn.Dense(7))(x)
        return x

    class Model(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = nn.WeightNorm(nn.Dense(3))(x)
        x = nn.Dense(4)(x)
        x = SubModel()(x)
        x = nn.Dense(8)(x)
        x = nn.SpectralNorm(nn.Dense(9))(x, update_stats=False)
        return x

    x = jnp.ones((1, 2))
    key = jax.random.key(0)
    model = Model()

    lines = model.tabulate(
      key,
      x,
      console_kwargs=CONSOLE_TEST_KWARGS,
      compute_flops=True,
      compute_vjp_flops=True,
    ).splitlines()

    self.assertIn('Model', lines[5])
    self.assertIn('WeightNorm_0', lines[7])
    self.assertIn('Dense_0/kernel/scale', lines[7])
    self.assertIn('Dense_0', lines[11])
    self.assertIn('Dense_1', lines[16])
    self.assertIn('SubModel_0', lines[21])
    self.assertIn('SubModel_0/SpectralNorm_0', lines[23])
    self.assertIn('Dense_0/kernel/sigma', lines[23])
    self.assertIn('Dense_0/kernel/u', lines[24])
    self.assertIn('SubModel_0/Dense_0', lines[28])
    self.assertIn('SubModel_0/Dense_1', lines[33])
    self.assertIn('SubModel_0/WeightNorm_0', lines[38])
    self.assertIn('Dense_2/kernel/scale', lines[38])
    self.assertIn('SubModel_0/Dense_2', lines[42])
    self.assertIn('Dense_2', lines[47])
    self.assertIn('SpectralNorm_0', lines[52])
    self.assertIn('Dense_3/kernel/sigma', lines[52])
    self.assertIn('Dense_3/kernel/u', lines[53])
    self.assertIn('Dense_3', lines[57])


if __name__ == '__main__':
  absltest.main()
