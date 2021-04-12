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
    error_page = 'https://flax.readthedocs.io/en/latest/flax.errors.html'
    module_name = self.__class__.__module__
    class_name = self.__class__.__name__
    error_msg = f'{message} ({error_page}#{module_name}.{class_name})'
    super().__init__(error_msg)


#################################################
# scope.py errors                               #
#################################################


class InvalidRngError(FlaxError):
  """
  All rngs used in a Module should be passed to 
  :meth:`Module.init() <flax.linen.Module.init>` and 
  :meth:`Module.apply() <flax.linen.Module.apply>` appropriately. We explain
  both separately using the following example::

    class Bar(nn.Module):
      @nn.compact
      def __call__(self, x):
        some_param = self.param('some_param', nn.initializers.zeros, (1, ))
        dropout_rng = self.make_rng('dropout')
        x = nn.Dense(features=4)(x)
        ...

    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = Bar()(x)
        ...

  **PRNGs for Module.init()**
  
  In this example, two rngs are used:

  * ``params`` is used for initializing the parameters of the model. This rng
    is used to initialize the ``some_params`` parameter, and for initializing
    the weights of the ``Dense`` Module used in ``Bar``.
  
  * ``dropout`` is used for the dropout rng that is used in ``Bar``.

  So, ``Foo`` is initialized as follows::
    
    init_rngs = {'params': random.PRNGKey(0), 'dropout': random.PRNGKey(1)}
    variables = Foo().init(init_rngs, init_inputs)

  If a Module only requires an rng for ``params``, you can use::

       SomeModule().init(rng, ...)  # Shorthand for {'params': rng}


  **PRNGs for Module.apply()**
  
  When applying ``Foo``, only the rng for ``dropout`` is needed, because 
  ``params`` is only used for initializing the Module parameters::

    Foo().apply(variables, inputs, rngs={'dropout': random.PRNGKey(2)})

  If a Module only requires an rng for ``params``, you don't have to provide
  rngs for apply at all::

       SomeModule().apply(variables, inputs)  # rngs=None
  """
  def __init__(self, msg):
    # For this error message we pass the entire message, since there are various
    # different kinds of RNG errors and we want to be able to be more specific
    # in the error message, while always linking to the same documentation.
    super().__init__(msg)


class ApplyScopeInvalidVariablesError(FlaxError):
  """
  When calling :meth:`Module.apply() <flax.linen.Module.apply>`, the first
  argument should be a variable dict. For more explanation on variable dicts,
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

    variables = Embed(4, 8).init(random.PRNGKey(0), jnp.ones((5, 5, 1)))
    _ = NoBiasDense().apply(variables, jnp.ones((5, 5, 1)), 'embed')
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

    variables = NoBiasDense().init(random.PRNGKey(0), jnp.ones((5, 5, 1)))
    _ = NoBiasDense().apply(variables, jnp.ones((5, 5)))
  """
  def __init__(self, param_name, scope_path, value_shape, init_shape):
    super().__init__('Inconsistent shapes between value and initializer '
                     f'for parameter "{param_name}" in "{scope_path}": '
                     f'{value_shape}, {init_shape}.')


class ScopeVariableNotFoundError(FlaxError):
  """
  This error is thrown when trying to use a variable in a Scope in a collection
  that is immutable. In order to create this variable, mark the collection as
  mutable explicitly using the ``mutable`` keyword in
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
    
    v = MyModule.init(...)
    ...
    logits = MyModule.apply(v, batch)  # This throws an error.
    logits = MyModule.apply(v, batch, mutable=['batch_stats'])  # This works.
  """
  def __init__(self, col, variable_name, scope_path):
    super().__init__(f'Cannot update variable "{variable_name}" in '
                     f'"{scope_path}" because collection "{col}" is immutable.')


#################################################
# module.py errors                              #
#################################################


class NameInUseError(FlaxError):
  """
  This error is raised when trying to create a submodule, param, or variable
  with an existing name. They are all considered to be in the same namespace.

  **Sharing Submodules**

  This is the wrong pattern for sharing submodules::

    y = nn.Dense(feature=3, name='bar')(x)
    z = nn.Dense(feature=3, name='bar')(x+epsilon)

  Instead, modules should be shared by instance::

    dense = nn.Dense(feature=3, name='bar')
    y = dense(x)
    z = dense(x+epsilon)

  If submodules are not provided with a name, a unique name will be given to
  them automatically::

    class MyModule(nn.Module):
      @nn.compact
      def __call__(self, x):
        x = MySubModule()(x)
        x = MySubModule()(x)  # This is fine.
        return x

  **Parameters and Variables**

  A parameter name can collide with a submodule or variable, since they are all
  stored in the same variable dict::

    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        bar = self.param('bar', nn.initializers.zeros, (1, ))
        embed = nn.Embed(num_embeddings=2, features=5, name='bar')  # <-- ERROR!
  
  Variables should also have unique names, even if they have their own
  collection::

    class Foo(nn.Module):
      @nn.compact
      def __call__(self, inputs):
        _ = self.param('mean', initializers.lecun_normal(), (2, 2))
        _ = self.variable('stats', 'mean', initializers.zeros, (2, 2))
  """
  def __init__(self, key_type, value, module_name):
    # key_type is in {param, variable, submodule}.
    super().__init__(f'Could not create {key_type} "{value}" in Module '
                     f'{module_name}: Name in use.')


class AssignSubModuleError(FlaxError):
  """
  You are only allowed to create submodules in two places:

  1.  If your Module is noncompact: inside 
      :meth:`Module.setup() <flax.linen.Module.setup>`.
  2.  If your Module is compact: inside the method wrapped in 
      :meth:`nn.compact() <flax.linen.compact>`.

  For instance, the following code throws this error, because ``nn.Conv`` is
  created in ``__call__``, which is not marked as compact::

    class Foo(nn.Module):
      def setup(self):
        pass

      def __call__(self, x):
        conv = nn.Conv(features=3, kernel_size=3)

    Foo().init(random.PRNGKey(0), jnp.zeros((1,)))

  Note that this error is also thrown if you partially defined a Module inside
  setup::

    class Foo(nn.Module):
      def setup(self):
        self.conv = functools.partial(nn.Conv, features=3)

      def __call__(self, x):
        x = self.conv(kernel_size=4)(x)
        return x

    Foo().init(random.PRNGKey(0), jnp.zeros((1,)))

  In this case, ``self.conv(kernel_size=4)`` is called from ``__call__``, which
  is disallowed beause it's neither within ``setup`` nor a method wrapped in
  x``nn.compact``.
  """
  def __init__(self, cls):
    super().__init__(f'Submodule {cls} must be defined in `setup()` or in a '
                     'method wrapped in `@compact`')


class SetAttributeInModuleSetupError(FlaxError):
  """
  You are not allowed to modify Module class attributes in
  :meth:`Module.setup() <flax.linen.Module.setup>`::

    class Foo(nn.Module):
      features: int = 6

      def setup(self):
        self.features = 3  # <-- ERROR

      def __call__(self, x):
        return nn.Dense(self.features)(x)

    variables = SomeModule().init(random.PRNGKey(0), jnp.ones((1, )))

  Instead, these attributes should be set when initializing the Module::

    class Foo(nn.Module):
      features: int = 6

      @nn.compact
      def __call__(self, x):
        return nn.Dense(self.features)(x)

    variables = SomeModule(features=3).init(random.PRNGKey(0), jnp.ones((1, )))
  
  TODO(marcvanzee): Link to a design note explaining why it's necessary for
  modules to stay frozen (otherwise we can't safely clone them, which we use for
  lifted transformations).
  """
  def __init__(self):
    super().__init__(f'Module construction attributes are frozen.')


class SetAttributeFrozenModuleError(FlaxError):
  """
  You can only assign Module attributes to ``self`` inside
  :meth:`Module.setup() <flax.linen.Module.setup>`. Outside of that method, the
  Module instance is frozen (i.e., immutable). This behavior is similar to
  frozen Python dataclasses.
  
  For instance, this error is raised in the following case::

    class SomeModule(nn.Module):
      @nn.compact
      def __call__(self, x, num_features=10):
        self.num_features = num_features  # <-- ERROR!
        x = nn.Dense(self.num_features)(x)
        return x

    s = SomeModule().init(random.PRNGKey(0), jnp.ones((5, 5)))

  Similarly, the error is raised when trying to modify a submodule's attributes
  after constructing it, even if this is done in the ``setup()`` method of the
  parent module::

    class Foo(nn.Module):
        def setup(self):
          self.dense = nn.Dense(features=10)
          self.dense.features = 20  # <--- This is not allowed
        
        def __call__(self, x):
          return self.dense(x)
  """
  def __init__(self, module_cls, attr_name, attr_val):
    super().__init__(f'Can\'t set {attr_name}={attr_val} for Module of type '
                    f'{module_cls}: Module instance is frozen outside of '
                     'setup method.')


class MultipleMethodsCompactError(FlaxError):
  """
  The ``@compact`` decorator may only be added to at most one method in a Flax
  module. In order to resolve this, you can:
  
  * remove ``@compact`` and define submodules and variables using 
    :meth:`Module.setup() <flax.linen.Module.setup>`.
  * Use two separate modules that both have a unique ``@compact`` method.

  TODO(marcvanzee): Link to a design note explaining the motivation behind this.
  There is no need for an equivalent to `hk.transparent` and it makes submodules
  much more sane because there is no need to prefix the method names.
  """
  def __init__(self):
    super().__init__(f'Only one method per class can be @compact')

class ReservedModuleAttributeError(FlaxError):
  """
  This error is thrown when creating a Module that is using reserved attributes.
  The following attributes are reserved:
  
  * ``parent``: The parent Module of this Module.
  * ``name``: The name of this Module.
  """
  def __init__(self, annotations):
    super().__init__(f'properties `parent` and `name` are reserved: '
                     f'{annotations}')


class ApplyModuleInvalidMethodError(FlaxError):
  """
  When calling :meth:`Module.apply() <flax.linen.Module.apply>`, you can specify
  the method to apply using parameter ``method``. This error is thrown if the
  provided parameter is not a method in the Module and not a function with at
  least one argument.

  Learn more on the reference docs for
  :meth:`Module.apply() <flax.linen.Module.apply>`.
  """
  def __init__(self, method):
    super().__init__(f'Cannot call apply(): {method} is not a valid function '
                     'for apply().')


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

    from jax import random
    variables = test_dense.init(random.PRNGKey(0), jnp.ones((5,5)))

    y = test_dense.apply(variables, jnp.ones((5,5)))
  """
  def __init__(self):
    super().__init__('Can\'t call compact methods on unbound modules')


class InvalidCheckpointError(FlaxError):
  """
  A checkpoint cannot be stored in a directory that already has
  a checkpoint at the current or a later step.

  You can pass `overwrite=True` to disable this behavior and
  overwrite existing checkpoints in the target directory.
  """
  def __init__(self, path, step):
    super().__init__(f'Trying to save an outdated checkpoint at step: "{step}" and path: "{path}".')
