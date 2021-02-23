- Start Date: 2021-02-08
- FLIP PR: [#1011](https://github.com/google/flax/pull/1011)
- FLIP Issue: [#1009](https://github.com/google/flax/issues/1009)

Table of contents:

- [Summary]
- [Motivation]
- [Functional API]
- [Using Optax]
- [Multi Opimizer]
- [Previous API]
- [Current Examples]
- [Linen Helper]

# Summary
[summary]: #summary

Our current API (referred to as [previous API] in this document) uses a pattern
where an `Optimizer` dataclass is created from a pytree of `target` variables
and from an `OptimizerDef` that defines how to update optimizer state,
hyperparameters, and target variables. This pattern is relatively complex for
implementing a simple optimizer, while being quite verbose in the typical Linen
train step (especially when using mutable state collections).

This FLIP proposes a new purely [functional API] where optimizers are simply
defined with a `init()` and an `update()` function and a new [Linen helper] that
reduces boilerplate in a typical Linen train step. See also the section
[Using Optax] for how to use Deepmind's [Optax] package, and the section about
[Multi Opimizer].

# Motivation
[motivation]: #motivation

Flax contains some optimizers in the `flax.optim` package, but list is far from
exhaustive and ideally we would instead use JAX optimizers from a dedicated PyPi
package.

Deepmind's [Optax] library is such a dedicated package that already implements a
choice of interesting optimizers and it also provides a framework to compose new
optimizers from reusable gradient transformations.

[Optax]: https://github.com/deepmind/optax

# Functional API
[Functional API]: #functional-api

An optimizer is defined by two pure functions:

```python
# flax/opt/momentum.py

@flax.struct.dataclass
class MomentumState:
  learning_rate: Optional[jnp.ndarray]
  beta: jnp.ndarray
  momentum: PyTree


def init(
    params: PyTree, learning_rate: Optional[float] = None, beta: float = 0.9
    ) -> MomentumState:
  opt_state = MomentumState(
      learning_rate=learning_rate,
      beta=beta,
      momentum=jax.tree_map(jnp.zeros_like, params),
  )
  return opt_state


def update(params: PyTree, grads: PyTree, opt_state: MomentumState
    ) -> Tuple[PyTree, MomentumState]:
  assert opt_state.learning_rate
  new_momentum = jax.tree_multimap(
      lambda momentum, grad: momentum * opt_state.beta + grad,
      opt_state.momentum, grads)
  new_params = jax.tree_multimap(
      lambda param, grad: param - opt_state.learning_rate * grad,
      params, grads)
  new_opt_state = opt_state.replace(
      momentum=new_momentum,
  )
  return new_params, new_opt_state
```

Alternatively, the `update()` function could be rewritten in a more terse form
using a new `@multimap` decorator that maps both a function's inputs and outputs
with a `jax.tree_multimap()`:

```python
def update(params: PyTree, grads: PyTree, opt_state: MomentumState
    ) -> Tuple[PyTree, MomentumState]:
  assert opt_state.learning_rate
  
  @multimap
  def inner(param, grad, momentum):
    new_momentum = momentum * opt_state.beta + grad
    new_param = param - opt_state.learning_rate * grad
    return new_param, new_momentum

  new_params, new_momentum = inner(params, grads, opt_state.momentum)

  return new_params, opt_state.replace(momentum=new_momentum)
```

Remarks:

- The main motivation to introduce this new API is its simplicity.
- See further down in [Linen helper] for an example of how to use this API in a
  typical Linen train step.

# Using Optax
[Using Optax]: #using-optax

# Multi Opimizer
[Multi Opimizer]: #multi-optimizer

# Previous API
[previous API]: #previous-api

An optimizer would first be constructed from its definition and the pytree of
target params:

```python
optimizer_def = flax.optim.Momentum(learning_rate=0.1, beta=0.9)
optimizer = optimizer_def.create(variables['params'])
```

The target variables would then optimized in the train step (assuming a single
non-params collection "batch_stats"):

```python
def make_train_step(apply_fn):
  @jax.jit
  def train_step(optimizer, batch_stats, inputs, labels):

    def loss_fn(params):
      variables = {'params': params, 'batch_stats': batch_stats}
      logits, new_model_state = apply_fn(
          variables, inputs, mutable=['batch_stats'])
      loss = xent_loss(logits, labels)
      return loss, new_model_state['batch_stats']

    (loss, new_batch_stats), grad = jax.value_and_grad(loss_fn, has_aux=True)(
        optimizer.target)
    lr = get_learning_rate(step)
    new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)
    return new_optimizer, new_batch_stats, loss

  return train_step


batch_stats = variables['batch_stats']
train_step = make_train_step(model.apply)
for step, batch in enumerate(ds)
  optimizer, batch_stats, loss = train_step(
      optimizer, batch_stats, batch['image'], batch['label'])
```

The optimizer itself would be implemented by creating a new class deriving
from `OpimizerDef`:

```python
# flax/optim/momentum.py

@flax.struct.dataclass
class _MomentumHyperParams:
  learning_rate: np.ndarray
  beta: np.ndarray


@flax.struct.dataclass
class _MomentumParamState:
  momentum: onp.ndarray


class Momentum(flax.optim.OptimizerDef):

  def __init__(self, learning_rate=None, beta=0.9):
    super().__init__(
      _MomentumHyperParams(learning_rate, beta)
    )

  def init_param_state(self, param):
    return _MomentumParamState(jnp.zeros_like(param))

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    del step
    assert hyper_params.learning_rate is not None
    new_momentum = state.momentum * hyper_params.beta + grad
    new_params = param - hyper_params.learning_rate * new_momentum
    return new_params, _MomentumParamState(new_momentum)
```

Remarks:

- The functions `init_param_state()` and `apply_param_gradient()` are called
  for every leaf in the params/grads pytree. This makes it possible to write the
  calculations directly without `jax.tree_multimap()`.
- The interface was defined in pre-Linen without the distinction of `params` vs.
  other collections in `variables` in mind. The original API was elegant because
  one only needed to pass around the optimizer, which included the parameters,
  optimizer state, optimizer hyperparameters, and a reference to the
  `OptimizerDef` to perform the param/state update.
- Note that the current implementation contains some additional functionality
  like the `MultiOptimizer` that is not explicitly considered here to keep this
  FLIP short, but will be added to the new [functional API] as well.


# Current Examples
[Current Examples]: #current-examples

| Example  |      Flax      |    Optax    |              Comments               |
| -------- | -------------- | ----------- | ----------------------------------- |
| imagenet | optim.Momentum | optax.trace | DynamicScale can be used unchanged. |
| mnist    | optim.Momentum | optax.trace |                                     |
| nlp_seq  | optim.Adam     | optax.adamw |                                     |
| pixelcnn | optim.Adam     | optax.adam  |                                     |
| ppo      | optim.Adam     | optax.adam  |                                     |
| seq2seq  | optim.Adam     | optax.adam  |                                     |
| vae      | optim.Adam     | optax.adam  |                                     |
| wmt      | optim.Adam     | optax.adamw |                                     |


# Linen Helper
[Linen helper]: #linen-helper

Using the [functional API] directly makes for the following code:

```python
def make_train_step(apply_fn, opt_update_fn):
  @jax.jit
  def train_step(variables, opt_state, inputs, labels):

    def loss_fn(params):
      logits, new_model_state = apply_fn(
          variables.copy(dict(params=params)), inputs, mutable=['batch_stats'])
      loss = xent_loss(logits, labels)
      return loss, new_model_state

    (loss, new_model_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        variables['params'])
    lr = get_learning_rate(step)
    new_params, new_opt_state = opt_update_fn(
        variables['params'], grads,
        opt_state.replace(learning_rate=get_learning_rate(step)))
    new_variables = variables.copy(dict(
        params=variables['params'], **new_model_state))
    return new_opt_state, new_variables, loss

  return train_step

opt_state = flax.opt.momentum.init(
    variables['params'], learning_rate=0.1, beta=0.9)
train_step = make_train_step(model.apply, flax.opt.momentum.update)
for step, batch in enumerate(ds.as_numpy_iterator()):
  optimizer, variables, loss = train_step(
      variables, opt_state, batch['image'], batch['label'])
```

Remarks:

- As opposed to the [previous API], we now can keep have the entire `variables` 
  including the `params` as an input and output to the function.
- Splitting `params` from `variables` is still necessary inside the train step
  because we only want to calculate gradients with respect to `params` and not
  the entire `variables`.

TODO@andsteing add a new pattern that simplifies the use of an optimizer using
the functional API in a train step with mutable state variables.