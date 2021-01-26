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
  pass

class InvalidScopeError(FlaxError):
  """
  A temporary Scope is only valid within the context in which it is created::

    with Scope(variables, rngs=rngs).temporary() as root:
      y = fn(root, *args, **kwargs)
      # Here root is valid.
    # Here root is invalid.
  """
  pass

class MutableCollectionError(FlaxError):
  """
  Collections that are immutable cannot be modified. When you are applying a
  Module, you should specify which variable collections are mutable::

    class MyModule(nn.Module):
      @nn.compact
      def __call__(self, x):
        ...
        var = self.variable('batch_stats', 'mean', ...)
        ...
    
    vars = MyModule.init(...)
    ...
    logits = MyModule.apply(vars, batch)  # This throws an error.
    logits = MyModule.apply(vars, batch, mutable=['batch_stats'])  # This works.
  """


class ScopeNamingError(FlaxError):
  """
  Scope names should be strings and unique within a subscope::

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
  pass

