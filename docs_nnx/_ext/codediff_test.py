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

"""Tests for codediff Sphinx extension."""

from absl.testing import parameterized
from codediff import CodeDiffParser


class CodeDiffTest(parameterized.TestCase):
  def test_parse(self):
    input_text = r"""@jax.jit #!
def get_initial_params(key):   #!
  init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
  initial_params = CNN().init(key, init_val)['params']
  extra_line
  return initial_params
---
@jax.pmap #!
def get_initial_params(key):
  init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
  initial_params = CNN().init(key, init_val)['params']
  return initial_params"""

    expected_table = """.. tab-set::\n  \n  .. tab-item:: Single device\n    \n    .. code-block:: python\n      :emphasize-lines: 1,2\n    \n      @jax.jit\n      def get_initial_params(key):\n        init_val = jnp.ones((1, 28, 28, 1), jnp.float32)\n        initial_params = CNN().init(key, init_val)['params']\n        extra_line\n        return initial_params\n      \n  .. tab-item:: Ensembling on multiple devices\n    \n    .. code-block:: python\n      :emphasize-lines: 1\n    \n      @jax.pmap\n      def get_initial_params(key):\n        init_val = jnp.ones((1, 28, 28, 1), jnp.float32)\n        initial_params = CNN().init(key, init_val)['params']\n        return initial_params"""

    expected_testcodes = [
      r"""@jax.jit #!
def get_initial_params(key):   #!
  init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
  initial_params = CNN().init(key, init_val)['params']
  extra_line
  return initial_params
""",
      r"""@jax.pmap #!
def get_initial_params(key):
  init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
  initial_params = CNN().init(key, init_val)['params']
  return initial_params""",
    ]

    title_left = 'Single device'
    title_right = 'Ensembling on multiple devices'

    actual_table, actual_testcodes = CodeDiffParser().parse(
      lines=input_text.split('\n'),
      title=f'{title_left}, {title_right}',
    )

    actual_table = '\n'.join(actual_table)
    actual_testcodes = ['\n'.join(testcode) for testcode, _ in actual_testcodes]

    self.assertEqual(expected_table, actual_table)
    self.assertEqual(expected_testcodes[0], actual_testcodes[0])
    self.assertEqual(expected_testcodes[1], actual_testcodes[1])

  @parameterized.parameters(
    {
      'input_text': r"""x = 1
  ---
  x = 2
""",
      'title': 'Tab 0, Tab1, Tab2',
      'groups': None,
      'error_msg': 'Expected 2 code separator\\(s\\) for 3 tab\\(s\\), but got 1 code separator\\(s\\) instead.',
    },
    {
      'input_text': r"""x = 1
  ---
  x = 2
  ---
  x = 3
  ---
  x = 4
""",
      'title': 'Tab 0, Tab1, Tab2',
      'groups': None,
      'error_msg': 'Expected 2 code separator\\(s\\) for 3 tab\\(s\\), but got 3 code separator\\(s\\) instead.',
    },
    {
      'input_text': r"""x = 1
  ---
  x = 2
  ---
  x = 3
""",
      'title': 'Tab 0, Tab1, Tab2',
      'groups': 'tab0, tab2',
      'error_msg': 'Expected 3 group assignment\\(s\\) for 3 tab\\(s\\), but got 2 group assignment\\(s\\) instead.',
    },
    {
      'input_text': r"""x = 1
  ---
  x = 2
  ---
  x = 3
""",
      'title': 'Tab 0, Tab1, Tab2',
      'groups': 'tab0, tab1, tab2, tab3',
      'error_msg': 'Expected 3 group assignment\\(s\\) for 3 tab\\(s\\), but got 4 group assignment\\(s\\) instead.',
    },
  )
  def test_parse_errors(self, input_text, title, groups, error_msg):
    with self.assertRaisesRegex(ValueError, error_msg):
      _, _ = CodeDiffParser().parse(
        lines=input_text.split('\n'),
        title=title,
        groups=groups,
      )
