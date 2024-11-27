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


from flax import nnx
from absl.testing import absltest


class TestContainers(absltest.TestCase):
  def test_unbox(self):
    x = nnx.Param(
      1,
      on_get_value=lambda c, x: x + 3,  # type: ignore
    )

    assert x.value == 4

  def test_on_set_value(self):
    x: nnx.Param[int] = nnx.Param(
      1,  # type: ignore
      on_set_value=lambda c, x: x + 7,  # type: ignore
    )
    x.value = 5

    assert x.raw_value == 12

  def test_module_unbox(self):
    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.x = nnx.Param(1, on_get_value=lambda c, x: x + 3)

    module = Foo()

    assert module.x.value == 4
    assert vars(module)['x'].raw_value == 1

  def test_module_box(self):
    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.x = nnx.Param(
          1,
          on_set_value=lambda c, x: x + 7,  # type: ignore
        )

    module = Foo()
    module.x.value = 5

    assert module.x.value == 12
    assert vars(module)['x'].raw_value == 12


if __name__ == '__main__':
  absltest.main()
