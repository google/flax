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
import jax.numpy as jnp


class TestContainers(absltest.TestCase):
  def test_unbox(self):
    class CustomParam(nnx.Param):
      def get_value(self, **kwargs):
        return super().get_value(**kwargs) + 3

    x = CustomParam(jnp.array(1))

    assert x[...] == 4

  def test_on_set_value(self):
    class CustomParam(nnx.Param):
      def set_value(self, value, **kwargs):
        super().set_value(value + 7, **kwargs)

    x = CustomParam(jnp.array(1))
    x[...] = 5

    assert x.get_raw_value() == 12

  def test_module_unbox(self):
    class CustomParam(nnx.Param):
      def get_value(self, **kwargs):
        return super().get_value(**kwargs) + 3

    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.x = CustomParam(1)

    module = Foo()

    assert module.x.get_value() == 4
    assert vars(module)['x'].get_raw_value() == 1

  def test_module_box(self):
    class CustomParam(nnx.Param):
      def set_value(self, value, **kwargs):
        super().set_value(value + 7, **kwargs)

    class Foo(nnx.Module):
      def __init__(self) -> None:
        self.x = CustomParam(jnp.array(1))

    module = Foo()
    module.x[...] = 5

    assert module.x[...] == 12
    assert vars(module)['x'][...] == 12


if __name__ == '__main__':
  absltest.main()
