# Copyright 2023 The Flax Authors.
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

"""Tools leveraging Fiddle to shard param_spec-based models easily.

To define a sharding policy, simply subclass ``ShardingPolicy`` and add
overloads for layers you care about::

  class TestSharding(spec_sharding.ShardingPolicy):
    "Shards in a way optimimal for testing."

  @TestSharding.overload(Linear)
  def _(c: fdl.Config[Linear]):
    c.kernel.sharding.features = 'model'
    c.kernel.sharding.input_features = 'data'

  @TestSharding.overload(Attention)
  def _(c: fdl.Config[Attention])
    c.qkv.sharding.stack = ''
    c.qkv.sharding.num_heads = 'model'
    c.qkv.sharding.embedding_size = ''
    c.qkv.sharding.d_model = 'replica'

  @TestSharding.overload(Transformer)
  def _(c: fdl.Config[Transformer]):
    # Override the `Linear` sharing for just the last layer.
    c.last_linear.kernel.sharding.input_features = 'replica'

To apply the sharding to a model configuration, simply call::

  config = ...
  TestSharding.shard(config)

To apply multiple different sharding strategies to different portions of a model
(e.g. a main model, but to re-shard a sub-model with a different strategy),
simply apply them in order, and also to whatever sub-portions of the config
structure as desired::

  TestSharding.shard(config)
  CustomModelReshard.shard(config.sub_image_model)

You can display all the sharding for every layer using standard Fiddle tools,
such as graphviz, and CLI-based printing.

Note: this should likely move out of the Flax repo to a separate dependency, as
Flax probably shouldn't take a dependency on Fiddle.
"""

import dataclasses
import inspect
from typing import Any, Callable, Type, TypeVar

import fiddle as fdl
from fiddle import daglish
from flax import linen as nn
from flax.linen.experimental.params import param


ModuleT = TypeVar('ModuleT', bound=nn.Module)
PerModuleShardingPolicy = Callable[[fdl.Config[ModuleT]], None]


class ShardingPolicyType(type):
  _overloads: dict[Type[nn.Module], PerModuleShardingPolicy[Any]]

  def __init__(cls, name, bases, dct):
    super().__init__(name, bases, dct)
    if '__doc__' not in dct:
      raise TypeError('Documentation for your sharding policy is missing.')
    cls._overloads = {}

  def policy_for(
      cls, module_type: Type[ModuleT],
  ) -> Callable[
      [PerModuleShardingPolicy[ModuleT]], PerModuleShardingPolicy[ModuleT]
  ]:
    if existing_overload := cls._overloads.get(module_type, None):
      raise TypeError(
          f'Duplicate overloads for {module_type}, previous '
          f'definition: {existing_overload.__module__}.'
          f'{existing_overload.__name__}'
      )

    def register(
        f: PerModuleShardingPolicy[ModuleT],
    ) -> PerModuleShardingPolicy[ModuleT]:
      cls._overloads[module_type] = f
      return f

    return register

  def shard(cls, config: fdl.Config[Any]) -> None:
    _post_order_traversal_of_all_params_with_fixups(config, cls._apply)

  def fallback_policy(cls, config: fdl.Config[Any]) -> None:
    pass

  def _apply(cls, config: fdl.Config[Any]) -> None:
    if not isinstance(config, fdl.Buildable):
      return
    fn_or_cls = fdl.get_callable(config)
    # TODO(saeta): Support ShardingPolicy inheritance.
    # TODO(saeta): Support nn.Module inheritance for sharding as well by walking
    # the MRO.
    policy_fn = cls._overloads.get(fn_or_cls, cls.fallback_policy)
    policy_fn(config)


class ShardingPolicy(metaclass=ShardingPolicyType):
  """Base class for sharding policies. Subclass this type to make a new one.

  Example::

    class SimpleSharding(ShardingPolicy):
      "My simple sharding policy!"

    SimpleSharding.overload(MyLayer)
    def _(config: fdl.Config[MyLayer]):
      config.kernel.input_dims = 'model'
      config.kernel.output_dims = 'data'

    model_config = ...
    SimpleSharding.shard(model_config)
  """


def setup_sharding(config: fdl.Buildable[Any]) -> None:
  """Prepares `config` for sharding customizations.

  This function is automatically called before `ShardingPolicy.shard`.

  This function is idempotent, and safe to call multiple times.

  Args:
    config: The config containing the model to be sharded.
  """
  _post_order_traversal_of_all_params_with_fixups(config, lambda x: None)


def _post_order_traversal_of_all_params_with_fixups(
    config: fdl.Buildable[Any],
    policy: Callable[[fdl.Config[Any]], None],
    memoized: bool = False,
) -> None:
  def traverse(node: Any, state: daglish.State) -> None:
    if isinstance(node, fdl.Buildable):
      fn_or_cls = fdl.get_callable(node)
      if inspect.isclass(fn_or_cls) and issubclass(
          fdl.get_callable(node), nn.Module
      ):
        _infer_param_spec_configs(node)
      # Go through all arguments, even if they don't yet have default values.
      params = set(node.__signature__.parameters.keys()).union(
          node.__arguments__.keys()
      )
      for param in params:
        child = getattr(node, param, None)
        if child:
          state.call(child, daglish.Attr(param))
    elif state.is_traversable(node):
      state.flattened_map_children(node)
    policy(node)

  traversal_cls = (
      daglish.MemoizedTraversal if memoized else daglish.BasicTraversal
  )
  traversal: daglish.Traversal = traversal_cls(traverse, config)
  traverse(config, traversal.initial_state())


def _infer_param_spec_configs(config: fdl.Buildable[ModuleT]) -> None:
  """Idempotently sets custom fdl.Configs based on ParamSpec schemas."""
  module_type = fdl.get_callable(config)
  assert issubclass(module_type, nn.Module)
  assert dataclasses.is_dataclass(module_type)
  for field in dataclasses.fields(module_type):
    if field.type != param.Param:
      continue
    # Note: config.__getattr__ throws ValueError when it's a default factory.
    try:
      has_value = hasattr(config, field.name)
    except ValueError:
      has_value = False
    if not has_value:
      spec_config = fdl.Config(param.Param)
      spec_config.sharding = fdl.Config(
          _TypesafeParamSpecShardingFactory(module_type, field.name))
      setattr(config, field.name, spec_config)


class _TypesafeParamSpecShardingFactory:
  """A callable producing a sharding spec, with a __signature__ property.

  It's very convenient, when maintaining configurations, to rely on Fiddle's
  eager error checking to verify the configuration is keeping up with the
  underlying code, without needing to call `fdl.build` (which might consume
  actual resources, or require running on exotic hardware). As a result, we
  need to supply an accurate `__signature__` to Fiddle (instead of just passing
  a `**kwargs`-style anonymous function). To support efficient serialization
  (e.g. pickle / cloudpickle) we eschew an anonymous lambda function or closure
  and instead define a simple class. This also conveniently enough gives us a
  place to attach this doc string, explaining what we're doing.
  """

  def __init__(self, module_type: Type[ModuleT], field: str):
    self.module_type = module_type
    self.field = field

  def __call__(self, **kwargs: str):
    return kwargs

  @property
  def axis_names(self):
    all_fields = dataclasses.fields(self.module_type)
    field = next(f for f in all_fields if f.name == self.field)
    schema = param.get_param_schema(field)
    assert schema is not None
    return schema.keys()

  @property
  def __signature__(self) -> inspect.Signature:
    param_kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    make_param = lambda name: inspect.Parameter(name, param_kind)
    parameters = [make_param(name) for name in self.axis_names]
    return inspect.Signature(parameters)

  def __repr__(self):
    schema_str = ', '.join(self.axis_names)
    return f'ParamSpecShardingFactory[{schema_str}]'
