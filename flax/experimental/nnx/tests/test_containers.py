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


from flax.experimental import nnx


class TestContainers:
  def test_unbox(self):
    x = nnx.Param(
      1,
      get_value_hooks=[lambda c, x: x + 1, lambda c, x: x * 2],  # type: ignore
    )

    assert x.value == 4

  def test_box(self):
    x: nnx.Param[int] = nnx.Param(
      1,  # type: ignore
      set_value_hooks=[lambda c, x: x + 1, lambda c, x: x * 2],  # type: ignore
    )
    x.value = 5

    assert x.raw_value == 12

  def test_module_unbox(self):
    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.x = nnx.Param(
          1, get_value_hooks=[lambda c, x: x + 1, lambda c, x: x * 2]
        )

    module = Foo()

    assert module.x.value == 4
    assert vars(module)['x'].raw_value == 1

  def test_module_box(self):
    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.x = nnx.Param(
          1, set_value_hooks=[lambda c, x: x + 1, lambda c, x: x * 2]
        )

    module = Foo()
    module.x.value = 5

    assert module.x.value == 12
    assert vars(module)['x'].raw_value == 12
