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
""""""  # Use an empty top-level docstring so Sphinx won't output the one below.
"""Flax error classes.

=== When to create a Flax error class?

If an error message requires more explanation than a one-liner, it is useful to
add it as a separate error class. This may lead to some duplication with 
existing documentation or docstrings, but it will provide users with more help
when they are debugging a problem. We can also point to existing documentation
from the error docstring directly.

=== How to name the error class?

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

#################################################
# scope.py errors                               #
#################################################

class InitScopeInvalidRngsError(FlaxError):
  """
  When initializing a Module with
  :meth:`Module.init() <flax.linen.Module.init>`, the first argument can be of
  two forms:

  1. A single PRNGKey. This is in case only one PRNGKey is needed to initialize
     the ``params`` collection. Note that this::

       SomeModule(...).init(jax.random.PRNGKey(0), ...)

     Is shorthand for::

       SomeModule(...).init({'params': jax.random.PRNGKey(0)}, ...)

  2. A directionary mapping collections to the PRNGKey to initialize them with.
     This is useful if the Module has more rngs than one for ``params``.
     
     For instance, suppose an ``EncoderDecoder`` Module that requires an RNG for
     decoding tokens based on a categorical probability distribution. Then a 
     typical call looks as follows::

        EncoderDecoder(...).init({'params': rng1, 'decode': rng2}, ...)

     Note that even though they may be used inside submodules, the rngs for the
     collections should be defined at the top-level. So the ``EncoderDecoder``
     module above may contain a submodule ``Decoder``, which then uses the 
     ``decode`` collection. The RNGs will be passed down to submodules
     automatically.
  """
  def __init__(self):
    super().__init__('First argument passed to an init function should be a '
                     '`jax.PRNGKey` or a dictionary mapping strings to '
                     '`jax.PRNGKey`.')


class ApplyScopeInvalidRngsError(FlaxError):
  """
  When applying a Module, the `rng` argument should be a dictionary mapping 
  collections to the PRNGKeys that are used when computing their new values.

  For instance, suppose an ``EncoderDecoder`` Module that requires an RNG for
  decoding tokens based on a categorical probability distribution. Then a 
  typical call to :meth:`Module.apply() <flax.linen.Module.apply>` looks as
  follows::

     EncoderDecoder(...).apply(params, ... {'decode': rng2}, ...)

  Remarks:

  * While :meth:`Module.init() <flax.linen.Module.init>` requires a rngs for
    the collection ``params``, this is not necessary when applying the module,
    because this collection is only use to initialize the model with.
  * Even though they may be used inside submodules, the rngs for the collections
    should be defined at the top-level. So the ``EncoderDecoder`` module above 
    may contain a submodule ``Decoder``, which then uses the ``decode``
    collection. The RNGs will be passed down to submodules automatically.
  """
  def __init__(self):
    super().__init__('rngs should be a dictionary mapping strings to '
                     '`jax.PRNGKey`.')
                   

class ApplyScopeInvalidVariablesError(FlaxError):
  """
  When calling :meth:`Module.apply() <flax.linen.Module.apply>`, the first
  argument should be a variable dict. For more explanation on variable direct,
  please see :mod:`flax.core.variables`.
  """
  def __init__(self):
    super().__init__('The first argument passed to an apply function should be '
                     'a dictionary of collections. Each collection should be a '
                     'dictionary with string keys.')


class ScopeParamNotFoundError(FlaxError):
  """
  This error is thrown when trying to access a parameter that does not exist.
  For instance, in the code below, the initialized embedding name 'embedding'
  does not match the apply name 'embed'::

    class Embed(nn.Module):
    num_embeddings: int
    features: int
      
    @nn.compact
    def __call__(self, inputs, embed_name='embedding'):
      inputs = inputs.astype('int32')
      embedding = self.param(embed_name,
                            lecun_normal(),
                            (self.num_embeddings, self.features))    
      return embedding[inputs]

  vars = Embed(4, 8).init(random.PRNGKey(0), jnp.ones((5, 5, 1)))
  print(jax.tree_map(lambda x : x.shape, vars))
  _ = NoBiasDense().apply(vars, jnp.ones((5, 5, 1)), 'embed')
  

  """
  def __init__(self, param_name, scope_path):
    super().__init__(f'No parameter named "{param_name}" exists in '
                     f'"{scope_path}".')


class ScopeParamShapeError(FlaxError):
  """
  This error is thrown when the shape of an existing parameter is different from
  the shape of the return value of the ``init_fn``. This can happen when the 
  shape provided during :meth:`Module.apply() <flax.linen.Module.apply>` is
  different from the one used when intializing the module.
  
  For instance, the following code throws this error because the apply shape 
  (``(5, 5, 1)``) is different from the init shape (``(5, 5``). As a result, the
  shape of the kernel during ``init`` is ``(1, 8)``, and the shape during 
  ``apply`` is ``(5, 8)``, which results in this error.::

      class NoBiasDense(nn.Module):
      features: int = 8

      @nn.compact
      def __call__(self, x):
        kernel = self.param('kernel',
                            lecun_normal(),
                            (x.shape[-1], self.features))  # <--- ERROR
        y = lax.dot_general(x, kernel,
                            (((x.ndim - 1,), (0,)), ((), ())))
        return y

    vars = NoBiasDense().init(random.PRNGKey(0), jnp.ones((5, 5, 1)))
    _ = NoBiasDense().apply(vars, jnp.ones((5, 5)))
  """
  def __init__(self, param_name, scope_path, value_shape, init_shape):
    super().__init__('Inconsistent shapes between value and initializer '
                     f'for parameter "{param_name}" in "{scope_path}": '
                     f'{value_shape}, {init_shape}.')


class ScopeVariableNotFoundError(FlaxError):
  """
  This error is thrown when trying to use a variable in a Scope in a collection
  that is immutable. In order to create this variable, mark the collection as
  mutable explicitly using the `mutable` keyword in
  :meth:`Module.apply() <flax.linen.Module.apply>`.
  """
  def __init__(self, name, col, scope_path):
    super().__init__(f'No Variable named "{name}" for collection "{col}" '
                     f'exists in "{scope_path}".')


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


class ModifyScopeVariableError(FlaxError):
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


class ScopeNameTypeError(FlaxError):
  """
  Scope names should be strings.
  """
  def __init__(self, scope_name):
    super().__init__(f'The type of scope "{scope_name}" should be string but '
                     f'it is {type(scope_name)}')


class ScopeNameInUseError(FlaxError):
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