# Copyright 2023 The Flax Authors.
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

import pytest

from flax.experimental import nnx


class TestContainers:
  def test_node_idenpotence(self):
    x = nnx.Variable(1)
    x = nnx.Variable(x)

    assert isinstance(x, nnx.Variable)

  def test_variable_idenpotence(self):
    x = nnx.Variable(1)
    x = nnx.Variable(x)

    assert isinstance(x, nnx.Variable)
    assert x.value == 1

  def test_variable_cannot_change_collection(self):
    x = nnx.Param(1)

    with pytest.raises(ValueError, match='is not compatible with return type'):
      x = nnx.BatchStat(x)

  def test_container_cannot_change_type(self):
    x = nnx.Variable(1)

    with pytest.raises(ValueError, match='is not compatible with return type'):
      x = nnx.Param(x)

    x = nnx.Param(2)

    with pytest.raises(ValueError, match='is not compatible with return type'):
      x = nnx.Variable(x)

  def test_unbox(self):
    x: nnx.Param[int] = nnx.Param(
      1,
      get_value_hooks=[lambda c, x: x + 1, lambda c, x: x * 2],  # type: ignore
    )

    assert x.get_value() == 4

  def test_box(self):
    x: nnx.Param[int] = nnx.Param(
      1,  # type: ignore
      set_value_hooks=[lambda c, x: x + 1, lambda c, x: x * 2],  # type: ignore
    )
    x.set_value(5)

    assert x.value == 12

  def test_module_unbox(self):
    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.x = nnx.Param(
          1, get_value_hooks=[lambda c, x: x + 1, lambda c, x: x * 2]
        )

    module = Foo()

    assert module.x == 4
    assert vars(module)['x'].value == 1

  def test_module_box(self):
    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.x = nnx.Param(
          1, set_value_hooks=[lambda c, x: x + 1, lambda c, x: x * 2]
        )

    module = Foo()
    module.x = 5

    assert module.x == 12
    assert vars(module)['x'].value == 12
