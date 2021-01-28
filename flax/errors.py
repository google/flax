# Copyright 2021 The Flax Authors.
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

# Copyright 2020 The Flax Authors.
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
"""Flax error classes."""


class FlaxError(Exception):
  def __init__(self, message):
    error_page = 'https://flax.readthedocs.io/en/improve-error/flax.errors.html'
    module_name = self.__class__.__module__
    class_name = self.__class__.__name__
    error_msg = f'{message} ({error_page}#{module_name}.{class_name})'
    super().__init__(error_msg)


class InvalidFilterError(FlaxError):
  """
  A filter should be either a boolean, a string or a container object.
  """
  def __init__(self, filter_like):
    super().__init__(f'Invalid Filter: "{filter_like}"')


class InvalidScopeError(FlaxError):
  """
  A temporary Scope is only valid within the context in which it is created::

    with Scope(variables, rngs=rngs).temporary() as root:
      y = fn(root, *args, **kwargs)
      # Here root is valid.
    # Here root is invalid.
  """
  def __init__(self, scope_name):
    super().__init__(f'The scope "{scope_name}" is no longer valid.')


class VariableModificationError(FlaxError):
  """
  You cannot update a variable if the collection it belongs to is immutable.
  When you are applying a Module, you should specify which variable 
  collections are mutable::

    class MyModule(nn.Module):
      @nn.compact
      def __call__(self, x):
        ...
        var = self.variable('batch_stats', 'mean', ...)
        var.value = ...
        ...
    
    vars = MyModule.init(...)
    ...
    logits = MyModule.apply(vars, batch)  # This throws an error.
    logits = MyModule.apply(vars, batch, mutable=['batch_stats'])  # This works.
  """
  def __init__(self, col, variable_name, scope_path):
    super().__init__(f'Cannot update variable "{variable_name}" in '
                     f'"{scope_path}" because collection "{col}" is immutable.')


# TODO(marcvanzee): Make sure this error is thrown in Modules and not in Scope
# and rename it to something like "ModuleNameTypeError".
class NameTypeError(FlaxError):
  """
  Scope names should be strings.
  """
  def __init__(self, scope_name):
    super().__init__(f'The type of scope "{scope_name}" should be string but '
                     f'it is {type(scope_name)}')


# TODO(marcvanzee): Make sure this error is thrown in Modules and not in Scope
# and rename it to something like "ModuleNameInUseError".
class NameInUseError(FlaxError):
  """
  Module names are unique within a subscope::

    class MyModule(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = MySubModule(name='m1')(x)
        x = MySubModule(name='m1')(x)  # This is not allowed.
        return x

  If submodules are not provided with a name, a unique name will be given to
  them automatically::

    class MyModule(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = MySubModule()(x)
        x = MySubModule()(x)  # This is fine.
        return x
  """
  def __init__(self, scope_name):
    super().__init__(f'Duplicate use of scope name: "{scope_name}"')