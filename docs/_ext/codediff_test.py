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

"""Tests for codediff Sphinx extension."""

from absl.testing import absltest

from codediff import CodeDiffParser


class CodeDiffTest(absltest.TestCase):

  def test_parse(self):

    input_text = r'''@jax.jit #!
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
  return initial_params'''

    expected_table = r'''+----------------------------------------------------------+----------------------------------------------------------+
| Single device                                            | Ensembling on multiple devices                           |
+----------------------------------------------------------+----------------------------------------------------------+
| .. code-block:: python                                   | .. code-block:: python                                   |
|   :emphasize-lines: 1,2                                  |   :emphasize-lines: 1                                    |
|                                                          |                                                          |
|   @jax.jit                                               |   @jax.pmap                                              |
|   def get_initial_params(key):                           |   def get_initial_params(key):                           |
|     init_val = jnp.ones((1, 28, 28, 1), jnp.float32)     |     init_val = jnp.ones((1, 28, 28, 1), jnp.float32)     |
|     initial_params = CNN().init(key, init_val)['params'] |     initial_params = CNN().init(key, init_val)['params'] |
|     extra_line                                           |     return initial_params                                |
|     return initial_params                                |                                                          |
+----------------------------------------------------------+----------------------------------------------------------+'''

    expected_testcode = r'''@jax.pmap #!
def get_initial_params(key):
  init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
  initial_params = CNN().init(key, init_val)['params']
  return initial_params'''

    title_left = 'Single device'
    title_right = 'Ensembling on multiple devices'

    actual_table, actual_testcode = CodeDiffParser().parse(
      lines=input_text.split('\n'),
      title_left=title_left,
      title_right=title_right)
    actual_table = '\n'.join(actual_table)
    actual_testcode = '\n'.join(actual_testcode)

    self.assertEqual(expected_table, actual_table)
    self.assertEqual(expected_testcode, actual_testcode)
