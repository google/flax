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
from __future__ import annotations

import functools
import typing as tp

import jax
from jax import random
import jax.numpy as jnp

from flax import struct
from flax import typing
from flax.nnx import graph
from flax.nnx.nn import initializers
from flax.nnx.variablelib import Variable
from flax.nnx import filterlib
from flax.nnx.pytreelib import Pytree
from flax.typing import MISSING, Key, Missing

F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
Counts = list[int]
AxesValue = tp.Union[int, None]
SplitPattern = tp.Union[AxesValue, tuple[AxesValue, ...]]
OutShardingType: tp.TypeAlias = (
  jax.sharding.NamedSharding | jax.sharding.PartitionSpec | None
)
Fargs = tp.ParamSpec('Fargs')


@tp.runtime_checkable
class KeylessInitializer(tp.Protocol):
  def __call__(
    self,
    shape: typing.Shape,
    dtype: tp.Any | None = None,
    out_sharding: OutShardingType = None,
  ) -> jax.Array:
    raise NotImplementedError


def _to_keyless(
  initializer_constructor: tp.Callable[Fargs, jax.nn.initializers.Initializer],
) -> tp.Callable[Fargs, KeylessInitializer]:
  raise NotImplementedError


def _function_to_method(random_f):
  def rngs_random_method(self: Rngs | RngStream, *args, **kwargs) -> jax.Array:
    return random_f(self(), *args, **kwargs)

  return rngs_random_method


def _initializer_to_method(
  initializer_constructor: tp.Callable[Fargs, jax.nn.initializers.Initializer],
):
  def rngs_initializer_method(
    self: Rngs | RngStream, *args: Fargs.args, **kwargs: Fargs.kwargs
  ) -> KeylessInitializer:
    init_fn = initializer_constructor(*args, **kwargs)

    def rngs_keyless_initializer(*init_args, **init_kwargs):
      return init_fn(self(), *init_args, **init_kwargs)

    return rngs_keyless_initializer

  return rngs_initializer_method


class RngState(Variable[jax.Array]):
  tag: str


class RngCount(RngState): ...


class RngKey(RngState): ...


NotKey = filterlib.All(RngState, filterlib.Not(RngKey))


class RngStream(Pytree):

  def __init__(
    self,
    key: jax.Array | int,
    *,
    tag: str,
  ):
    if isinstance(key, int):
      key = random.key(key)
    elif isinstance(key, jax.Array) and key.dtype == jnp.uint32:
      key = random.wrap_key_data(key)

    if not isinstance(key, jax.Array) or not jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
      raise ValueError(f'Invalid rng value: {key}, expected a '
                       f'jax.Array of jax.dtypes.prng_key sub-dtype')

    count = jnp.zeros(key.shape, dtype=jnp.uint32)
    self.tag = tag
    self.key = RngKey(key, tag=tag)
    self.count = RngCount(count, tag=tag)

  def __call__(self) -> jax.Array:
    self.count._check_can_update()
    key = random.fold_in(self.key[...], self.count[...])
    self.count[...] += 1
    return key

  def fork(self, *, split: int | tuple[int, ...] | None = None):
    key = self()
    if split is not None:
      key = random.split(key, split)
    return type(self)(key, tag=self.tag)

  # ----------------------------------------------------------
  # random functions
  # ----------------------------------------------------------
  if tp.TYPE_CHECKING:
    bits = staticmethod(functools.partial(random.bits, random.key(0)))
    uniform = staticmethod(
      functools.partial(random.uniform, random.key(0))
    )
    randint = staticmethod(
      functools.partial(random.randint, random.key(0))
    )
    permutation = staticmethod(
      functools.partial(random.permutation, random.key(0))
    )
    choice = staticmethod(functools.partial(random.choice, random.key(0)))
    normal = staticmethod(functools.partial(random.normal, random.key(0)))
    multivariate_normal = staticmethod(
      functools.partial(random.multivariate_normal, random.key(0))
    )
    truncated_normal = staticmethod(
      functools.partial(random.truncated_normal, random.key(0))
    )
    bernoulli = staticmethod(
      functools.partial(random.bernoulli, random.key(0))
    )
    beta = staticmethod(functools.partial(random.beta, random.key(0)))
    cauchy = staticmethod(functools.partial(random.cauchy, random.key(0)))
    dirichlet = staticmethod(
      functools.partial(random.dirichlet, random.key(0))
    )
    exponential = staticmethod(
      functools.partial(random.exponential, random.key(0))
    )
    gamma = staticmethod(functools.partial(random.gamma, random.key(0)))
    loggamma = staticmethod(
      functools.partial(random.loggamma, random.key(0))
    )
    poisson = staticmethod(
      functools.partial(random.poisson, random.key(0))
    )
    gumbel = staticmethod(functools.partial(random.gumbel, random.key(0)))
    categorical = staticmethod(
      functools.partial(random.categorical, random.key(0))
    )
    laplace = staticmethod(
      functools.partial(random.laplace, random.key(0))
    )
    logistic = staticmethod(
      functools.partial(random.logistic, random.key(0))
    )
    pareto = staticmethod(functools.partial(random.pareto, random.key(0)))
    t = staticmethod(functools.partial(random.t, random.key(0)))
    chisquare = staticmethod(
      functools.partial(random.chisquare, random.key(0))
    )
    f = staticmethod(functools.partial(random.f, random.key(0)))
    rademacher = staticmethod(
      functools.partial(random.rademacher, random.key(0))
    )
    maxwell = staticmethod(
      functools.partial(random.maxwell, random.key(0))
    )
    double_sided_maxwell = staticmethod(
      functools.partial(random.double_sided_maxwell, random.key(0))
    )
    weibull_min = staticmethod(
      functools.partial(random.weibull_min, random.key(0))
    )
    orthogonal = staticmethod(
      functools.partial(random.orthogonal, random.key(0))
    )
    generalized_normal = staticmethod(
      functools.partial(random.generalized_normal, random.key(0))
    )
    ball = staticmethod(functools.partial(random.ball, random.key(0)))
    rayleigh = staticmethod(
      functools.partial(random.rayleigh, random.key(0))
    )
    wald = staticmethod(functools.partial(random.wald, random.key(0)))
    geometric = staticmethod(
      functools.partial(random.geometric, random.key(0))
    )
    triangular = staticmethod(
      functools.partial(random.triangular, random.key(0))
    )
    lognormal = staticmethod(
      functools.partial(random.lognormal, random.key(0))
    )
    binomial = staticmethod(
      functools.partial(random.binomial, random.key(0))
    )
    multinomial = staticmethod(
      functools.partial(random.multinomial, random.key(0))
    )
  else:
    bits = _function_to_method(random.bits)
    uniform = _function_to_method(random.uniform)
    randint = _function_to_method(random.randint)
    permutation = _function_to_method(random.permutation)
    choice = _function_to_method(random.choice)
    normal = _function_to_method(random.normal)
    multivariate_normal = _function_to_method(random.multivariate_normal)
    truncated_normal = _function_to_method(random.truncated_normal)
    bernoulli = _function_to_method(random.bernoulli)
    beta = _function_to_method(random.beta)
    cauchy = _function_to_method(random.cauchy)
    dirichlet = _function_to_method(random.dirichlet)
    exponential = _function_to_method(random.exponential)
    gamma = _function_to_method(random.gamma)
    loggamma = _function_to_method(random.loggamma)
    poisson = _function_to_method(random.poisson)
    gumbel = _function_to_method(random.gumbel)
    categorical = _function_to_method(random.categorical)
    laplace = _function_to_method(random.laplace)
    logistic = _function_to_method(random.logistic)
    pareto = _function_to_method(random.pareto)
    t = _function_to_method(random.t)
    chisquare = _function_to_method(random.chisquare)
    f = _function_to_method(random.f)
    rademacher = _function_to_method(random.rademacher)
    maxwell = _function_to_method(random.maxwell)
    double_sided_maxwell = _function_to_method(random.double_sided_maxwell)
    weibull_min = _function_to_method(random.weibull_min)
    orthogonal = _function_to_method(random.orthogonal)
    generalized_normal = _function_to_method(random.generalized_normal)
    ball = _function_to_method(random.ball)
    rayleigh = _function_to_method(random.rayleigh)
    wald = _function_to_method(random.wald)
    geometric = _function_to_method(random.geometric)
    triangular = _function_to_method(random.triangular)
    lognormal = _function_to_method(random.lognormal)
    binomial = _function_to_method(random.binomial)
    multinomial = _function_to_method(random.multinomial)

  # ----------------------------------------------------------
  # initializers
  # ----------------------------------------------------------
  if tp.TYPE_CHECKING:
    # skip constant
    delta_orthogonal = staticmethod(_to_keyless(initializers.delta_orthogonal))
    glorot_normal = staticmethod(_to_keyless(initializers.glorot_normal))
    glorot_uniform = staticmethod(_to_keyless(initializers.glorot_uniform))
    he_normal = staticmethod(_to_keyless(initializers.he_normal))
    he_uniform = staticmethod(_to_keyless(initializers.he_uniform))
    kaiming_normal = staticmethod(_to_keyless(initializers.kaiming_normal))
    kaiming_uniform = staticmethod(_to_keyless(initializers.kaiming_uniform))
    lecun_normal = staticmethod(_to_keyless(initializers.lecun_normal))
    lecun_uniform = staticmethod(_to_keyless(initializers.lecun_uniform))
    # skip normal as it conflicts with jax.random.normal
    # skip ones
    # skip orthogonal as it conflicts with jax.random.orthogonal
    # skip truncated_normal as it conflicts with jax.random.truncated_normal
    # skip uniform as it conflicts with jax.random.uniform
    variance_scaling = staticmethod(_to_keyless(initializers.variance_scaling))
    xavier_normal = staticmethod(_to_keyless(initializers.xavier_normal))
    xavier_uniform = staticmethod(_to_keyless(initializers.xavier_uniform))
    # skip zeros
  else:
    # skip constant
    delta_orthogonal = _initializer_to_method(initializers.delta_orthogonal)
    glorot_normal = _initializer_to_method(initializers.glorot_normal)
    glorot_uniform = _initializer_to_method(initializers.glorot_uniform)
    he_normal = _initializer_to_method(initializers.he_normal)
    he_uniform = _initializer_to_method(initializers.he_uniform)
    kaiming_normal = _initializer_to_method(initializers.kaiming_normal)
    kaiming_uniform = _initializer_to_method(initializers.kaiming_uniform)
    lecun_normal = _initializer_to_method(initializers.lecun_normal)
    lecun_uniform = _initializer_to_method(initializers.lecun_uniform)
    # skip normal as it conflicts with jax.random.normal
    # skip ones
    # skip orthogonal as it conflicts with jax.random.orthogonal
    # skip truncated_normal as it conflicts with jax.random.truncated_normal
    # skip uniform as it conflicts with jax.random.uniform
    variance_scaling = _initializer_to_method(initializers.variance_scaling)
    xavier_normal = _initializer_to_method(initializers.xavier_normal)
    xavier_uniform = _initializer_to_method(initializers.xavier_uniform)
    # skip zeros


RngValue = tp.Union[int, jax.Array]

class Rngs(Pytree):
  """A small abstraction to manage RNG state.

  ``Rngs`` allows the creation of ``RngStream`` which are used to easily generate new unique
  random keys on demand. An ``RngStream`` is a wrapper around a JAX random ``key``, and a
  ``counter``. Every time a key is requested, the counter is incremented and the key is
  generated from the seed key and the counter by using ``jax.random.fold_in``.

  To create an ``Rngs`` pass in an integer or ``jax.random.key`` to the
  constructor as a keyword argument with the name of the stream. The key will be used as the
  starting seed for the stream, and the counter will be initialized to zero. Then call the
  stream to get a key::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> rngs = nnx.Rngs(params=0, dropout=1)

    >>> param_key1 = rngs.params()
    >>> param_key2 = rngs.params()
    >>> dropout_key1 = rngs.dropout()
    >>> dropout_key2 = rngs.dropout()
    ...
    >>> assert param_key1 != dropout_key1

  Trying to generate a key for a stream that was not specified during construction
  will result in an error being raised::

    >>> rngs = nnx.Rngs(params=0, dropout=1)
    >>> try:
    ...   key = rngs.unkown_stream()
    ... except AttributeError as e:
    ...   print(e)
    No RngStream named 'unkown_stream' found in Rngs.

  The ``default`` stream can be created by passing in a key to the constructor without
  specifying a stream name. When the ``default`` stream is set the ``rngs`` object can be
  called directly to get a key, and calling streams that were not specified during
  construction will fallback to ``default``::

    >>> rngs = nnx.Rngs(0, params=1)
    ...
    >>> key1 = rngs.default()       # uses 'default'
    >>> key2 = rngs()               # uses 'default'
    >>> key3 = rngs.params()        # uses 'params'
    >>> key4 = rngs.dropout()       # uses 'default'
    >>> key5 = rngs.unkown_stream() # uses 'default'
  """

  def __init__(
    self,
    default: RngValue
    | RngStream
    | tp.Mapping[str, RngValue | RngStream]
    | None = None,
    **rngs: RngValue | RngStream,
  ):
    """
    Args:
      default: the starting seed for the ``default`` stream, defaults to None.
      **rngs: keyword arguments specifying the starting seed for each stream.
        The key can be an integer or a ``jax.random.key``.
    """
    if default is not None:
      if isinstance(default, tp.Mapping):
        rngs = {**default, **rngs}
      else:
        rngs['default'] = default

    for tag, key in rngs.items():
      if isinstance(key, RngStream):
        key = key.key.get_value()
      stream = RngStream(
        key=key,
        tag=tag,
      )
      setattr(self, tag, stream)

  def _get_stream(self, name: str, error_type: type[Exception]) -> RngStream:
    stream_vars = vars(self)
    if name not in stream_vars:
      if 'default' not in stream_vars:
        raise error_type(f"No RngStream named '{name}' found in Rngs.")
      stream = stream_vars['default']
    else:
      stream = stream_vars[name]

    return stream

  def __getitem__(self, name: str):
    return self._get_stream(name, KeyError)

  def __getattr__(self, name: str):
    return self._get_stream(name, AttributeError)

  def __call__(self):
    return self.default()

  def __iter__(self) -> tp.Iterator[str]:
    for name, stream in vars(self).items():
      if isinstance(stream, RngStream):
        yield name

  def __len__(self) -> int:
    return sum(
      1 for stream in vars(self).values() if isinstance(stream, RngStream)
    )

  def __contains__(self, name: tp.Any) -> bool:
    return name in vars(self)

  def items(self):
    for name, stream in vars(self).items():
      if isinstance(stream, RngStream):
        yield name, stream

  def fork(
    self,
    /,
    *,
    split: tp.Mapping[filterlib.Filter, int | tuple[int, ...]]
    | int
    | tuple[int, ...]
    | None = None,
  ):
    """Returns a new Rngs object with new unique RNG keys.

    Example::
      >>> from flax import nnx
      ...
      >>> rngs = nnx.Rngs(params=1, dropout=2)
      >>> new_rngs = rngs.fork()
      ...
      >>> assert rngs.params() != new_rngs.params()

    ``split`` can be used to split the keys of the newly created ``Rngs`` object::

      >>> rngs = nnx.Rngs(params=1, dropout=2)
      >>> new_rngs = rngs.fork(split=5)
      ...
      >>> assert new_rngs.params.key.shape == (5,)
      >>> assert new_rngs.dropout.key.shape == (5,)

    ``split`` also accepts a mapping of
    `Filters <https://flax.readthedocs.io/en/latest/guides/filters_guide.html>`__  to
    split sizes or None to control which streams are split and how they are split::

      >>> rngs = nnx.Rngs(params=1, dropout=2, noise=3)
      >>> new_rngs = rngs.fork(split={
      ...  'params': 5,      # split params into 5 keys
      ...  'dropout': None,  # don't split dropout
      ...  ...: (2, 5),      # split anything else into 2x5 keys
      ... })
      ...
      >>> assert new_rngs.params.key.shape == (5,)
      >>> assert new_rngs.dropout.key.shape == ()
      >>> assert new_rngs.noise.key.shape == (2, 5)
    """
    if split is None:
      split = {}
    elif isinstance(split, int):
      split = {...: split}
    elif isinstance(split, tuple):
      split = {...: split}

    split_predicates = {filterlib.to_predicate(k): v for k, v in split.items()}
    keys: dict[str, RngStream] = {}
    for name, stream in self.items():
      for predicate, num_splits in split_predicates.items():
        if predicate((), stream):
          keys[name] = stream.fork(split=num_splits)
          break
      else:
        keys[name] = stream.fork()

    return Rngs(**keys)

  # ----------------------------------------------------------
  # random functions
  # ----------------------------------------------------------
  if tp.TYPE_CHECKING:
    bits = staticmethod(functools.partial(random.bits, random.key(0)))
    uniform = staticmethod(
      functools.partial(random.uniform, random.key(0))
    )
    randint = staticmethod(
      functools.partial(random.randint, random.key(0))
    )
    permutation = staticmethod(
      functools.partial(random.permutation, random.key(0))
    )
    choice = staticmethod(functools.partial(random.choice, random.key(0)))
    normal = staticmethod(functools.partial(random.normal, random.key(0)))
    multivariate_normal = staticmethod(
      functools.partial(random.multivariate_normal, random.key(0))
    )
    truncated_normal = staticmethod(
      functools.partial(random.truncated_normal, random.key(0))
    )
    bernoulli = staticmethod(
      functools.partial(random.bernoulli, random.key(0))
    )
    beta = staticmethod(functools.partial(random.beta, random.key(0)))
    cauchy = staticmethod(functools.partial(random.cauchy, random.key(0)))
    dirichlet = staticmethod(
      functools.partial(random.dirichlet, random.key(0))
    )
    exponential = staticmethod(
      functools.partial(random.exponential, random.key(0))
    )
    gamma = staticmethod(functools.partial(random.gamma, random.key(0)))
    loggamma = staticmethod(
      functools.partial(random.loggamma, random.key(0))
    )
    poisson = staticmethod(
      functools.partial(random.poisson, random.key(0))
    )
    gumbel = staticmethod(functools.partial(random.gumbel, random.key(0)))
    categorical = staticmethod(
      functools.partial(random.categorical, random.key(0))
    )
    laplace = staticmethod(
      functools.partial(random.laplace, random.key(0))
    )
    logistic = staticmethod(
      functools.partial(random.logistic, random.key(0))
    )
    pareto = staticmethod(functools.partial(random.pareto, random.key(0)))
    t = staticmethod(functools.partial(random.t, random.key(0)))
    chisquare = staticmethod(
      functools.partial(random.chisquare, random.key(0))
    )
    f = staticmethod(functools.partial(random.f, random.key(0)))
    rademacher = staticmethod(
      functools.partial(random.rademacher, random.key(0))
    )
    maxwell = staticmethod(
      functools.partial(random.maxwell, random.key(0))
    )
    double_sided_maxwell = staticmethod(
      functools.partial(random.double_sided_maxwell, random.key(0))
    )
    weibull_min = staticmethod(
      functools.partial(random.weibull_min, random.key(0))
    )
    orthogonal = staticmethod(
      functools.partial(random.orthogonal, random.key(0))
    )
    generalized_normal = staticmethod(
      functools.partial(random.generalized_normal, random.key(0))
    )
    ball = staticmethod(functools.partial(random.ball, random.key(0)))
    rayleigh = staticmethod(
      functools.partial(random.rayleigh, random.key(0))
    )
    wald = staticmethod(functools.partial(random.wald, random.key(0)))
    geometric = staticmethod(
      functools.partial(random.geometric, random.key(0))
    )
    triangular = staticmethod(
      functools.partial(random.triangular, random.key(0))
    )
    lognormal = staticmethod(
      functools.partial(random.lognormal, random.key(0))
    )
    binomial = staticmethod(
      functools.partial(random.binomial, random.key(0))
    )
    multinomial = staticmethod(
      functools.partial(random.multinomial, random.key(0))
    )
  else:
    bits = _function_to_method(random.bits)
    uniform = _function_to_method(random.uniform)
    randint = _function_to_method(random.randint)
    permutation = _function_to_method(random.permutation)
    choice = _function_to_method(random.choice)
    normal = _function_to_method(random.normal)
    multivariate_normal = _function_to_method(random.multivariate_normal)
    truncated_normal = _function_to_method(random.truncated_normal)
    bernoulli = _function_to_method(random.bernoulli)
    beta = _function_to_method(random.beta)
    cauchy = _function_to_method(random.cauchy)
    dirichlet = _function_to_method(random.dirichlet)
    exponential = _function_to_method(random.exponential)
    gamma = _function_to_method(random.gamma)
    loggamma = _function_to_method(random.loggamma)
    poisson = _function_to_method(random.poisson)
    gumbel = _function_to_method(random.gumbel)
    categorical = _function_to_method(random.categorical)
    laplace = _function_to_method(random.laplace)
    logistic = _function_to_method(random.logistic)
    pareto = _function_to_method(random.pareto)
    t = _function_to_method(random.t)
    chisquare = _function_to_method(random.chisquare)
    f = _function_to_method(random.f)
    rademacher = _function_to_method(random.rademacher)
    maxwell = _function_to_method(random.maxwell)
    double_sided_maxwell = _function_to_method(random.double_sided_maxwell)
    weibull_min = _function_to_method(random.weibull_min)
    orthogonal = _function_to_method(random.orthogonal)
    generalized_normal = _function_to_method(random.generalized_normal)
    ball = _function_to_method(random.ball)
    rayleigh = _function_to_method(random.rayleigh)
    wald = _function_to_method(random.wald)
    geometric = _function_to_method(random.geometric)
    triangular = _function_to_method(random.triangular)
    lognormal = _function_to_method(random.lognormal)
    binomial = _function_to_method(random.binomial)
    multinomial = _function_to_method(random.multinomial)

  # ----------------------------------------------------------
  # initializers
  # ----------------------------------------------------------
  if tp.TYPE_CHECKING:
    # skip constant
    delta_orthogonal = staticmethod(_to_keyless(initializers.delta_orthogonal))
    glorot_normal = staticmethod(_to_keyless(initializers.glorot_normal))
    glorot_uniform = staticmethod(_to_keyless(initializers.glorot_uniform))
    he_normal = staticmethod(_to_keyless(initializers.he_normal))
    he_uniform = staticmethod(_to_keyless(initializers.he_uniform))
    kaiming_normal = staticmethod(_to_keyless(initializers.kaiming_normal))
    kaiming_uniform = staticmethod(_to_keyless(initializers.kaiming_uniform))
    lecun_normal = staticmethod(_to_keyless(initializers.lecun_normal))
    lecun_uniform = staticmethod(_to_keyless(initializers.lecun_uniform))
    # skip normal as it conflicts with jax.random.normal
    # skip ones
    # skip orthogonal as it conflicts with jax.random.orthogonal
    # skip truncated_normal as it conflicts with jax.random.truncated_normal
    # skip uniform as it conflicts with jax.random.uniform
    variance_scaling = staticmethod(_to_keyless(initializers.variance_scaling))
    xavier_normal = staticmethod(_to_keyless(initializers.xavier_normal))
    xavier_uniform = staticmethod(_to_keyless(initializers.xavier_uniform))
    # skip zeros
  else:
    # skip constant
    delta_orthogonal = _initializer_to_method(initializers.delta_orthogonal)
    glorot_normal = _initializer_to_method(initializers.glorot_normal)
    glorot_uniform = _initializer_to_method(initializers.glorot_uniform)
    he_normal = _initializer_to_method(initializers.he_normal)
    he_uniform = _initializer_to_method(initializers.he_uniform)
    kaiming_normal = _initializer_to_method(initializers.kaiming_normal)
    kaiming_uniform = _initializer_to_method(initializers.kaiming_uniform)
    lecun_normal = _initializer_to_method(initializers.lecun_normal)
    lecun_uniform = _initializer_to_method(initializers.lecun_uniform)
    # skip normal as it conflicts with jax.random.normal
    # skip ones
    # skip orthogonal as it conflicts with jax.random.orthogonal
    # skip truncated_normal as it conflicts with jax.random.truncated_normal
    # skip uniform as it conflicts with jax.random.uniform
    variance_scaling = _initializer_to_method(initializers.variance_scaling)
    xavier_normal = _initializer_to_method(initializers.xavier_normal)
    xavier_uniform = _initializer_to_method(initializers.xavier_uniform)
    # skip zeros


StreamBackup = (
  tuple[RngStream, jax.Array, jax.Array] | tuple[RngStream, jax.Array]
)


class SplitBackups(struct.PyTreeNode, tp.Iterable[StreamBackup]):
  backups: list[StreamBackup]

  def __iter__(self) -> tp.Iterator[StreamBackup]:
    return iter(self.backups)

  def __enter__(self):
    return self

  def __exit__(self, *args):
    restore_rngs(self)


@tp.overload
def split_rngs(
  node: tp.Any,
  /,
  *,
  splits: int | tuple[int, ...],
  only: filterlib.Filter = ...,
  squeeze: bool = False,
) -> SplitBackups: ...
@tp.overload
def split_rngs(
  *,
  splits: int | tuple[int, ...],
  only: filterlib.Filter = ...,
  squeeze: bool = False,
) -> tp.Callable[[F], F]: ...
def split_rngs(
  node: tp.Any = MISSING,
  /,
  *,
  splits: int | tuple[int, ...],
  only: filterlib.Filter = ...,
  squeeze: bool = False,
) -> SplitBackups | tp.Callable[[F], F]:
  """Splits the (nested) Rng states of the given node.

  Args:
    node: the base node containing the rng states to split.
    splits: an integer or tuple of integers specifying the
      shape of the split rng keys.
    only: a Filter selecting which rng states to split.

  Returns:
    A SplitBackups iterable if ``node`` is provided, otherwise a
    decorator that splits the rng states of the inputs to the
    decorated function.

  Example::

    >>> from flax import nnx
    ...
    >>> rngs = nnx.Rngs(params=0, dropout=1)
    >>> _ = nnx.split_rngs(rngs, splits=5)
    >>> rngs.params.key.shape, rngs.dropout.key.shape
    ((5,), (5,))

    >>> rngs = nnx.Rngs(params=0, dropout=1)
    >>> _ = nnx.split_rngs(rngs, splits=(2, 5))
    >>> rngs.params.key.shape, rngs.dropout.key.shape
    ((2, 5), (2, 5))


    >>> rngs = nnx.Rngs(params=0, dropout=1)
    >>> _ = nnx.split_rngs(rngs, splits=5, only='params')
    >>> rngs.params.key.shape, rngs.dropout.key.shape
    ((5,), ())

  Once split, random state can be used with transforms like :func:`nnx.vmap`::

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...     self.dropout = nnx.Dropout(0.5, rngs=rngs)
    ...
    >>> rngs = nnx.Rngs(params=0, dropout=1)
    >>> _ = nnx.split_rngs(rngs, splits=5, only='params')
    ...
    >>> state_axes = nnx.StateAxes({(nnx.Param, 'params'): 0, ...: None})
    ...
    >>> @nnx.vmap(in_axes=(state_axes,), out_axes=state_axes)
    ... def create_model(rngs):
    ...   return Model(rngs)
    ...
    >>> model = create_model(rngs)
    >>> model.dropout.rngs.key.shape
    ()

  ``split_rngs`` returns a SplitBackups object that can be used to restore the
  original unsplit rng states using :func:`nnx.restore_rngs`, this is useful
  when you only want to split the rng states temporarily::

    >>> rngs = nnx.Rngs(params=0, dropout=1)
    ...
    >>> backups = nnx.split_rngs(rngs, splits=5, only='params')
    >>> model = create_model(rngs)
    >>> nnx.restore_rngs(backups)
    ...
    >>> model.dropout.rngs.key.shape
    ()

  SplitBackups can also be used as a context manager to automatically restore
  the rng states when exiting the context::

    >>> rngs = nnx.Rngs(params=0, dropout=1)
    ...
    >>> with nnx.split_rngs(rngs, splits=5, only='params'):
    ...   model = create_model(rngs)
    ...
    >>> model.dropout.rngs.key.shape
    ()

    >>> state_axes = nnx.StateAxes({(nnx.Param, 'params'): 0, ...: None})
    ...
    >>> @nnx.split_rngs(splits=5, only='params')
    ... @nnx.vmap(in_axes=(state_axes,), out_axes=state_axes)
    ... def create_model(rngs):
    ...   return Model(rngs)
    ...
    >>> rngs = nnx.Rngs(params=0, dropout=1)
    >>> model = create_model(rngs)
    >>> model.dropout.rngs.key.shape
    ()


  """
  if isinstance(node, Missing):

    def split_rngs_decorator(f: F) -> F:
      @functools.wraps(f)
      def split_rngs_wrapper(*args, **kwargs):
        with split_rngs(
          (args, kwargs), splits=splits, only=only, squeeze=squeeze
        ):
          return f(*args, **kwargs)

      return tp.cast(F, split_rngs_wrapper)

    return split_rngs_decorator  # type: ignore[bad-return-type]

  if squeeze and splits != 1:
    raise ValueError('squeeze=True is only supported for splits=1')

  predicate = filterlib.to_predicate(only)
  backups: list[StreamBackup] = []
  for path, stream in graph.iter_graph(node):
    if (
      isinstance(stream, RngStream)
      and predicate((*path, 'key'), stream.key)
      and predicate((*path, 'count'), stream.count)
    ):
      key = stream()
      backups.append((stream, stream.key[...], stream.count[...]))
      key = random.split(key, splits)
      if squeeze:
        key = key[0]
      stream.key.set_value(key)
      if squeeze:
        counts_shape = stream.count.shape
      elif isinstance(splits, int):
        counts_shape = (splits, *stream.count.shape)
      else:
        counts_shape = (*splits, *stream.count.shape)

      stream.count.set_value(jnp.zeros(counts_shape, dtype=jnp.uint32))

  return SplitBackups(backups)

@tp.overload
def fork_rngs(
  node: tp.Any,
  /,
  *,
  split: tp.Mapping[filterlib.Filter, int | tuple[int, ...] | None]
    | int
    | None = None,
) -> SplitBackups: ...
@tp.overload
def fork_rngs(
  *,
  split: tp.Mapping[filterlib.Filter, int | tuple[int, ...] | None]
    | int
    | None = None,
) -> tp.Callable[[F], F]: ...
def fork_rngs(
  node: tp.Any = MISSING,
  /,
  *,
  split: tp.Mapping[filterlib.Filter, int | tuple[int, ...] | None]
    | int
    | None = None,
) -> SplitBackups | tp.Callable[[F], F]:
  """Forks the (nested) Rng states of the given node.

  Args:
    node: the base node containing the rng states to fork.
    split: an integer, tuple of integers, or mapping specifying the
      shape of the forked rng keys. If a mapping, keys are filters selecting
      which rng states to fork with the corresponding split shape.

  Returns:
    A SplitBackups iterable if ``node`` is provided, otherwise a
    decorator that forks the rng states of the inputs to the
    decorated function.

  Example::

    >>> from flax import nnx
    ...
    >>> rngs = nnx.Rngs(params=0, dropout=1)
    >>> _ = nnx.fork_rngs(rngs, split=5)
    >>> rngs.params.key.shape, rngs.dropout.key.shape
    ((5,), (5,))

    >>> rngs = nnx.Rngs(params=0, dropout=1)
    >>> _ = nnx.fork_rngs(rngs, split=(2, 5))
    >>> rngs.params.key.shape, rngs.dropout.key.shape
    ((2, 5), (2, 5))


    >>> rngs = nnx.Rngs(params=0, dropout=1)
    >>> _ = nnx.fork_rngs(rngs, split={'params': 5})
    >>> rngs.params.key.shape, rngs.dropout.key.shape
    ((5,), ())

  Once forked, random state can be used with transforms like :func:`nnx.vmap`::

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...     self.dropout = nnx.Dropout(0.5, rngs=rngs)
    ...
    >>> rngs = nnx.Rngs(params=0, dropout=1)
    >>> _ = nnx.fork_rngs(rngs, split={'params': 5})
    ...
    >>> state_axes = nnx.StateAxes({(nnx.Param, 'params'): 0, ...: None})
    ...
    >>> @nnx.vmap(in_axes=(state_axes,), out_axes=state_axes)
    ... def create_model(rngs):
    ...   return Model(rngs)
    ...
    >>> model = create_model(rngs)
    >>> model.dropout.rngs.key.shape
    ()

  ``fork_rngs`` returns a SplitBackups object that can be used to restore the
  original unforked rng states using :func:`nnx.restore_rngs`, this is useful
  when you only want to fork the rng states temporarily::

    >>> rngs = nnx.Rngs(params=0, dropout=1)
    ...
    >>> backups = nnx.fork_rngs(rngs, split={'params': 5})
    >>> model = create_model(rngs)
    >>> nnx.restore_rngs(backups)
    ...
    >>> model.dropout.rngs.key.shape
    ()

  SplitBackups can also be used as a context manager to automatically restore
  the rng states when exiting the context::

    >>> rngs = nnx.Rngs(params=0, dropout=1)
    ...
    >>> with nnx.fork_rngs(rngs, split={'params': 5}):
    ...   model = create_model(rngs)
    ...
    >>> model.dropout.rngs.key.shape
    ()

    >>> state_axes = nnx.StateAxes({(nnx.Param, 'params'): 0, ...: None})
    ...
    >>> @nnx.fork_rngs(split={'params': 5})
    ... @nnx.vmap(in_axes=(state_axes,), out_axes=state_axes)
    ... def create_model(rngs):
    ...   return Model(rngs)
    ...
    >>> rngs = nnx.Rngs(params=0, dropout=1)
    >>> model = create_model(rngs)
    >>> model.dropout.rngs.key.shape
    ()
  """
  if isinstance(node, Missing):

    def fork_rngs_decorator(f: F) -> F:
      @functools.wraps(f)
      def fork_rngs_wrapper(*args, **kwargs):
        with fork_rngs((args, kwargs), split=split):
          return f(*args, **kwargs)

      return tp.cast(F, fork_rngs_wrapper)

    return fork_rngs_decorator  # type: ignore[bad-return-type]

  if split is None:
    split = {...: None}
  elif isinstance(split, int | tuple):
    split = {...: split}

  predicate_splits = {
    filterlib.to_predicate(k): v for k, v in split.items()
  }
  backups: list[StreamBackup] = []
  for path, stream in graph.iter_graph(node):
    for predicate, splits in predicate_splits.items():
      if (
        isinstance(stream, RngStream)
        and predicate((*path, 'key'), stream.key)
        and predicate((*path, 'count'), stream.count)
      ):
        forked_stream = stream.fork(split=splits)
        # backup the original stream state
        backups.append((stream, stream.key[...], stream.count[...]))
        # apply the forked key and count to the original stream
        stream.key.set_value(forked_stream.key.get_value())
        stream.count.set_value(forked_stream.count.get_value())

  return SplitBackups(backups)


def backup_keys(node: tp.Any, /):
  backups: list[StreamBackup] = []
  for _, stream in graph.iter_graph(node):
    if isinstance(stream, RngStream):
      backups.append((stream, stream.key[...]))
  return backups

def _scalars_only(
  path: tuple[Key, ...], scalar_key: jax.Array, target_shape: tuple[int, ...]
) -> jax.Array:
  if target_shape != ():
    raise ValueError(
      f'Cannot reseed stream at path {path!r} becuase it has a non-scalar key, '
      f'found key with shape {target_shape}. If all your multi-dimensional '
      'keys have unique values on all dimensions, set policy="match_shape", '
      'else provide a custom reseed policy.'
    )
  return scalar_key


def _match_shape(
  path: tuple[Key, ...], scalar_key: jax.Array, target_shape: tuple[int, ...]
) -> jax.Array:
  if target_shape == ():
    return scalar_key
  return random.split(scalar_key, target_shape)


def reseed(
  node,
  /,
  *,
  policy: tp.Literal['scalars_only', 'match_shape']
  | tp.Callable[
    [tuple, jax.Array, tuple[int, ...]], jax.Array
  ] = 'scalars_only',
  **stream_keys: RngValue,
):
  """Update the keys of the specified RNG streams with new keys.

  Args:
    node: the node to reseed the RNG streams in.
    policy: defines how the the new scalar key is for each RngStream is used to
      reseed the stream. If ``'scalars_only'`` is given (the default), an error is raised
      if the target stream key is not a scalar. If ``'match_shape'`` is given, the new
      scalar key is split to match the shape of the target stream key. A callable
      of the form ``(path, scalar_key, target_shape) -> new_key`` can be passed to
      define a custom reseeding policy.
    **stream_keys: a mapping of stream names to new keys. The keys can be
      either integers or ``jax.random.key``.

  Example::

    >>> from flax import nnx
    >>> import jax.numpy as jnp
    ...
    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...     self.dropout = nnx.Dropout(0.5, rngs=rngs)
    ...   def __call__(self, x):
    ...     return self.dropout(self.linear(x))
    ...
    >>> model = Model(nnx.Rngs(params=0, dropout=42))
    >>> x = jnp.ones((1, 2))
    ...
    >>> y1 = model(x)
    ...
    >>> # reset the ``dropout`` stream key to 42
    >>> nnx.reseed(model, dropout=42)
    >>> y2 = model(x)
    ...
    >>> jnp.allclose(y1, y2)
    Array(True, dtype=bool)
  """
  if policy == 'scalars_only':
    policy = _scalars_only
  elif policy == 'match_shape':
    policy = _match_shape
  elif not callable(policy):
    raise ValueError(
      f'policy must be "scalars_only", "match_shape" or a callable, '
      f'got {policy!r}'
    )
  rngs = Rngs(**stream_keys)
  for path, stream in graph.iter_graph(node):
    if isinstance(stream, RngStream):
      if stream.key.tag in stream_keys:
        key = rngs[stream.key.tag]()
        key = policy(path, key, stream.key.shape)
        stream.key.set_value(key)
        stream.count.set_value(jnp.zeros(key.shape, dtype=jnp.uint32))


def restore_rngs(backups: tp.Iterable[StreamBackup], /):
  for backup in backups:
    stream = backup[0]
    stream.key.set_value(backup[1])
    if len(backup) == 3:
      stream.count.set_value(backup[2])  # count
