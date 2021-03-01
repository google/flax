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
"""Flax error classes.

=== Suggested error classes naming conventions:

* If the error occurs when doing something, name the error
  <Verb><Object><TypeOfError>Error

  For instance, if you want to raise an error when applying a module with an 
  invalid method, the error can be: ApplyModuleInvalidMethodError.

 <TypeOfError> is optional, for instance if there is only one error when 
  modifying a variable, the error can simply be: ModifyVariableError.

* If there is no concrete action involved the only a description of the error is
  sufficient. For instance: InvalidFilterError, NameInUseError, etc.


=== Copy/pastable template for new error messages:

class Template(FlaxError):
  "" "

  "" "
  def __init__(self):
    super().__init__(f'')
"""

class FlaxError(Exception):
  def __init__(self, message):
    error_page = 'https://flax.readthedocs.io/en/improve-error/flax.errors.html'
    module_name = self.__class__.__module__
    class_name = self.__class__.__name__
    error_msg = f'{message} ({error_page}#{module_name}.{class_name})'
    super().__init__(error_msg)


class AssignSubModuleOutsideSetupError(FlaxError):
  """


  """
  def __init__(self):
    super().__init__(f'You can only assign submodules to self in setup().')


class SetAttributeFrozenModuleError(FlaxError):
  """
  You can only assign Flax Module attributes to `self` inside the
  :meth:`Module.setup() <flax.linen.Module.setup>` method. Outside of that 
  method, the Module instance is frozen (i.e., immutable), meaning you can't
  modify it. This behavior is similar to frozen Python dataclasses.
  
  For instance, this error is raised in the following case::

  class SomeModule(nn.Module):
    @nn.compact
    def __call__(self, x, num_features=10):
      self.num_features = num_features
      x = nn.Dense(self.num_features)(x)
      return x

  s = SomeModule().init(random.PRNGKey(0), jnp.ones((5, 5)))

  This error is also thrown if you try to 
  
  """
  def __init__(self):
    super().__init__(f'Module instance is frozen outside of setup method.')


class MultipleMethodsCompactError(FlaxError):
  """
  The ``@compact`` decorator may only be added to at most one method in a Flax
  module. In order to resolve this, you can:
  
  * remove ``@compact`` and define submodules and variables using 
    :meth:`Module.setup() <flax.linen.Module.setup>`.
  * Use two separate modules that both have a unique ``@compact`` method.
  """
  def __init__(self):
    super().__init__(f'Only one method per class can be @compact')

class ReservedModuleAttributeError(FlaxError):
  """
  This error is thrown when creating a Flax Module that is using reserved
  attributes. The following attributes are reserved:
  
  * parent: The parent Module of this Module.
  * name: The name of this Module.
  """
  def __init__(self, annotations):
    super().__init__(f'properties `parent` and `name` are reserved: '
                     f'{annotations}')


class InitModuleInvalidRngError(FlaxError):
  """
  This error is thrown if the RNG you provide to one of the ``init`` functions
  of a Flax MOdule has an incorrect shape (it should be ``(2,)``).

  When initializing a Module with :meth:`Module.init() <flax.linen.Module.init>`
  or :meth:`Module.init_with_output() <flax.linen.Module.init_with_output>`, you
  should provide the RNGs required for initializing all variable collections as
  the first argument in a dictionary::

    from jax import random
    rngs = random.split(random.PRNGKey(0), n)
    rng_dict = {"collection_1": rngs[0], ..., "collection_n": rng[n-1]}
    vars = SomeModule.init(rng_dict, ...)
  
  Often, a Module only has collection "params", in which case it is allowed to
  just provide the RNG as the first argument to the ``init`` function::

    rng = jax.random.PRNGKey(0)
    vars = SomeModule.init(rng, ...)

  This error is thrown is you provide a single RNG, but it is not of shape
  ``(2,)``.
  """
  def __init__(self, module_name, rngs):
    super().__init__(f'RNGs should be of shape (2,) in Module {module_name}, '
                     f'but rngs are: {rngs}')


class ApplyModuleInvalidMethodError(FlaxError):
  """
  When calling :meth:`Module.apply() <flax.linen.Module.apply>`, you can specify
  the method to apply with parameters `method`. This error is thrown if the
  provided parameter is not a method..
  """
  def __init__(self, module_name, method):
    super().__init__(f'Cannot call apply() for {module_name}: {method} is not a '
                     'method.')


class CallCompactUnboundModuleError(FlaxError):
  """
  This error occurs when you are trying to call a Module directly, rather than
  through :meth:`Module.apply() <flax.linen.Module.apply>`. For instance, the 
  error will be raised when trying to run this code::

    from flax import linen as nn
    import jax.numpy as jnp

    test_dense = nn.Dense(10)
    test_dense(jnp.ones((5,5)))

  Instead, you should pass the variables (parameters and other state) via 
  :meth:`Module.apply() <flax.linen.Module.apply>` (or use 
  :meth:`Module.init() <flax.linen.Module.init>` to get initial variables)::

    # Create the initialized variables with a random key for inits:
    from jax import random
    vars = test_dense.init(random.PRNGKey(0), jnp.ones((5,5)))

    # Apply the NN to the variables + input to get output.
    y = test_dense.apply(vars, jnp.ones((5,5)))


  """
  def __init__(self):
    super().__init__('Can\'t call compact methods on unbound modules')


class JaxOmnistagingError(FlaxError):
  """
  The Flax linen API requires JAX omnistaging to be enabled. In order to enable
  this, add this to your imports::
    
    from jax.config import config
    config.enable_omnistaging()
  """
  def __init(self):
    super().__init__(f'Flax Linen requires Omnistaging to be enabled')


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


class ModifyVariableError(FlaxError):
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