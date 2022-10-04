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

"""Tests for kw_only_dataclasses."""

import dataclasses
import inspect
from absl.testing import absltest

from flax.linen import kw_only_dataclasses


class KwOnlyDataclassesTest(absltest.TestCase):

  def test_kwonly_args_moved_to_end(self):

    @kw_only_dataclasses.dataclass
    class TestClass:
      a: int = 1
      b: int = kw_only_dataclasses.field(default=2, kw_only=True)
      c: int = 3

    params = inspect.signature(TestClass.__init__).parameters
    self.assertEqual(list(params), ['self', 'a', 'c', 'b'])
    self.assertEqual(params['a'].default, 1)
    self.assertEqual(params['b'].default, 2)
    self.assertEqual(params['c'].default, 3)

    v1 = TestClass()
    self.assertDictEqual(dataclasses.asdict(v1), dict(a=1, b=2, c=3))

    v2 = TestClass(b=20)
    self.assertDictEqual(dataclasses.asdict(v2), dict(a=1, b=20, c=3))

    v3 = TestClass(1, 30)
    self.assertDictEqual(dataclasses.asdict(v3), dict(a=1, b=2, c=30))

  def test_base_optional_subclass_required(self):

    @kw_only_dataclasses.dataclass
    class Parent:
      a: int = kw_only_dataclasses.field(default=2, kw_only=True)

    @kw_only_dataclasses.dataclass
    class Child(Parent):
      b: int

    child_params = inspect.signature(Child.__init__).parameters
    self.assertEqual(list(child_params), ['self', 'b', 'a'])
    self.assertEqual(child_params['a'].default, 2)
    self.assertEqual(child_params['b'].default, inspect.Parameter.empty)

    v1 = Child(4)
    self.assertDictEqual(dataclasses.asdict(v1), dict(a=2, b=4))

    v2 = Child(4, 5)  # pylint: disable=too-many-function-args
    self.assertDictEqual(dataclasses.asdict(v2), dict(a=5, b=4))

  def test_subclass_overrides_base(self):
    # Note: if a base class declares a field as keyword-only, then
    # subclasses don't need to also declare it as keyword-only.

    @kw_only_dataclasses.dataclass
    class A:
      x: int = kw_only_dataclasses.field(default=1, kw_only=True)

    @kw_only_dataclasses.dataclass
    class B(A):
      size: float
      y: int = kw_only_dataclasses.field(default=3, kw_only=True)
      x: int = 2

    @kw_only_dataclasses.dataclass
    class C(B):
      name: str

    a_params = inspect.signature(A.__init__).parameters
    b_params = inspect.signature(B.__init__).parameters
    c_params = inspect.signature(C.__init__).parameters

    self.assertEqual(list(a_params), ['self', 'x'])
    self.assertEqual(list(b_params), ['self', 'size', 'x', 'y'])
    self.assertEqual(list(c_params), ['self', 'size', 'name', 'x', 'y'])

    self.assertEqual(a_params['x'].default, 1)
    self.assertEqual(b_params['x'].default, 2)
    self.assertEqual(b_params['y'].default, 3)
    self.assertEqual(b_params['size'].default, inspect.Parameter.empty)
    self.assertEqual(c_params['x'].default, 2)
    self.assertEqual(c_params['y'].default, 3)
    self.assertEqual(c_params['name'].default, inspect.Parameter.empty)
    self.assertEqual(c_params['size'].default, inspect.Parameter.empty)

    value = C(4, 'foo')  # pylint: disable=too-many-function-args
    self.assertDictEqual(
        dataclasses.asdict(value), dict(name='foo', size=4, x=2, y=3))

  def test_kwonly_marker(self):

    @kw_only_dataclasses.dataclass
    class A:
      x: float
      _: kw_only_dataclasses.KW_ONLY
      a: int = 5
      b: int = kw_only_dataclasses.field(default=2)
      c: int = kw_only_dataclasses.field(default=2, kw_only=True)

    @kw_only_dataclasses.dataclass
    class B(A):
      z: str

    a_params = inspect.signature(A.__init__).parameters
    b_params = inspect.signature(B.__init__).parameters
    self.assertEqual(list(a_params), ['self', 'x', 'a', 'b', 'c'])
    self.assertEqual(list(b_params), ['self', 'x', 'z', 'a', 'b', 'c'])


if __name__ == '__main__':
  absltest.main()
