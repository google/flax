- Start Date: 2021-02-08
- FLIP PR: [#1011](https://github.com/google/flax/pull/1011)
- FLIP Issue: [#1009](https://github.com/google/flax/issues/1009)

Table of contents:

- [Summary]
- [Motivation]
- [Functional API]
  - [Functional Implementation]
  - [Using Optax]
  - [Multi Optimizer]
  - [Functional Usage]
  - [Linen Helper]
- [Previous API]
  - [Previous Implementation]
  - [Previous Usage]
- [Update Plan]


# Summary
[Summary]: #summary

Our current API (referred to as [previous API] in this document) uses a pattern
where an `Optimizer` dataclass is created from a pytree of `target` variables
and from an `OptimizerDef` that defines how to update optimizer state,
hyperparameters, and target variables. This pattern is relatively complex for
implementing a simple optimizer, while being quite verbose in the typical Linen
train step (especially when using mutable state collections).

This FLIP proposes a new purely [functional API] where optimizers are simply
defined by an `init()` and an `update()` function. This new pattern allows
[using Optax] optimizers that follow a very similar functional pattern.

In the end, this document also proposes a new small [Linen helper] that takes
care of the usual splitting and merging of variables for optimization.


# Motivation
[motivation]: #motivation

Flax contains some optimizers in the `flax.optim` package, but that list is far
from exhaustive and ideally we would instead use JAX optimizers from a dedicated
PyPi package.

DeepMind already has a dedicated library — [Optax] — that implements a wide
range of interesting optimizers and provides a framework to compose new
optimizers from reusable gradient transformations.

[Optax]: https://github.com/deepmind/optax


# Functional API
[Functional API]: #functional-api

## Functional Implementation
[Functional Implementation]: #functional-implementation

An optimizer is defined by two pure functions. Since we will be [using Optax]
we define the `Optimizer` as a named tuple (instead of a
`flax.struct.dataclass`):

```python
# flax/opt/base.py

Params = Any
Gradients = Any
OptimizerState = NamedTuple


class Optimizer(NamedTuple):
  init: Callable[[Params], OptimizerState]
  update: Callable[
      [Gradients, Params, OptimizerState], Tuple[Params, OptimizerState]]
```

For example, to implement `flax.optim.Momentum` in this new interface:

```python
# flax/opt/momentum.py

class MomentumState(OptimizerState):
  momentum: PyTree


def momentum(learning_rate: float, beta: float = 0.9) -> Optimizer:

  def init_fn(params: Params) -> OptimizerState:
    return MomentumState(momentum=jax.tree_map(jnp.zeros_like, params))

  def update_fn(
    grads: Gradients, params: Params, state: OptimizerState
  ) -> Tuple[Params, OptimizerState]:
    new_momentum = jax.tree_multimap(
        lambda momentum, grad: momentum * beta + grad,
        opt_state.momentum, grads)
    new_params = jax.tree_multimap(
        lambda param, grad: param - learning_rate * grad,
        params, grads)
    return new_params, MomentumState(new_momentum)

  return Optimizer(init_fn, update_fn)
```

Note that unlike in the [previous API], the function `update_fn()` will be
called directly in the train loop and must take care of processing the pytrees
of parameters and optimizer state with a `jax.tree_multimap()`. For more
complex optimizers this can become a hassle but we can provide a small
functional helper `@multimap` that takes care of this:

```python
  def update_fn(
    grads: Gradients, params: Params, state: OptimizerState
  ) -> Tuple[Params, OptimizerState]:

    @multimap
    def inner(param, grad, momentum):
      new_momentum = momentum * beta + grad
      new_param = param - learning_rate * grad
      return new_param, new_momentum

    new_params, new_momentum = inner(params, grads, state.momentum)
    return new_params, MomentumState(new_momentum)
```

Remarks:

- We'll call `Optimizer` instances from the functional API `optim`, and refer
  to `flax.optim.Optimizer` instances from the [previous API] as `optimizer`
  for the remainder of this document.
- Compared with the [previous API], this new API does not have the option to
  update hyper parameters. With Optax, updating the hyper parameters is instead
  done by defining a schedule that takes care of updating the hyper parameters
  during training (see the [using Optax] code snippet for an example).
- Using this simple API directly makes the user code a bit more verbose than
  using the [previous API], but the difference is smaller with Linen. If you
  care about a boilerplate-free training loop see the proposed [Linen helper]
  further down.

## Using Optax
[Using Optax]: #using-optax

Optax is centered on the idea of composable gradient transformations. Since
above-mentioned [functional API] is based on Optax's interface it is
straightforward to wrap an Optax transformation to comply with the `Optimizer`
interface:

```python
import optax


def optax_optimizer(tx: optax.GradientTransformation) -> Optimizer:
  """Wraps an Optax transformation to update parameters directly."""

  def update_fn(
    grads: Gradients, params: Params, state: OptimizerState
  ) -> Tuple[Params, OptimizerState]:
    updates, state = tx.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return params, state

  return Optimizer(tx.init, update_fn)

tx = optax.chain(
    optax.trace(decay=0.9, nesterov=False),
    optax.scale_by_schedule(lambda step: -get_learning_rate(step))
)

optim = optax_optimizer(tx)
```

Remarks:

- The only difference is that the Optax udpate function returns a transformed
  gradient and the updated optimizer state (which makes them composable), so we
  apply this to the parameters in a final step and return the updated parameters
  and optimizer state.
- We can use a `get_learning_rate()` that returns the learning rate depending on
  the step number when defining the Optax gradient update transformation. Above
  code illustrates how this can be a drop-in replacement for [previous Code],
  where this update function already exists (notice how we need to invert the
  sign because we add the gradient update to the parameters). See also [Optax
  #20](https://github.com/deepmind/optax/issues/20) for an ongoing discussion
  about scheduling hyper parameters.

## Multi Optimizer
[Multi Optimizer]: #multi-optimizer

The [previous API] defined `flax.optim.MultiOptimizer` for processing different
parts of the parameter tree with different optimizers:

```python
biases_traversal = flax.optim.ModelParamTraversal(
    lambda path, _: path.endswith('/bias'))
not_biases_traversal = flax.optim.ModelParamTraversal(
    lambda path, _: not path.endswith('/bias'))

optimizer_def = flax.optim.MultiOptimizer(
    (biases_traversal, flax.optim.GradientDescent(learning_rate=0.1)),
    (not_biases_traversal, flax.optim.GradientDescent(learning_rate=0.05)),
)
```

Note how we first define a traversal that selects parameters based on their
path (which is the concatenation of module scopes and variable name), and then
create a `MultiOptimizer` that binds a different optimizer for each of these
separate traversals.

A similar scheme could be implemented with optimizers following the functional
API:

```python
# flax/opt/base.py

FilterFn = Callable[[str], bool]


def multi_optimizer(
  *filters_and_optimizers: Tuple[FilterFn, Optimizer]
) -> Optimizer:
  """Each `Optimizer` is restricted to params where `FilterFn` returns True."""

  traversals_and_optimizers = [
      (flax.optim.ModelParamTraversal(lambda path, _: filter_fn(path)), opt)
      for filter_fn, opt in filters_and_optimizers
  ]

  def init_fn(params: Params) -> Sequence[OptimizerState]:
    return [
        opt.init(list(traversal.iterate(params)))
        for traversal, opt in traversals_and_optimizers
    ]

  def update_fn(
      grads: Gradients, params: Params, states: Sequence[OptimizerState]
  ) -> Tuple[Params, Sequence[OptimizerState]]:
    new_params = params
    new_states = []
    for (traversal, opt), state in zip(traversals_and_optimizers, states):
      p = list(traversal.iterate(params))
      g = list(traversal.iterate(grads))
      new_p, new_state = opt.update(g, p, state)
      new_params = traversal.set(new_p, new_params)
      new_states.append(new_state)
    return new_params, new_states

  return Optimizer(init_fn, update_fn)
```

Remarks:

- The `flax.optim.ModelParamTraversal` is reused.
- The interface is a bit simplified, allowing to filter by path only (filtering
  by value was never used), and shortening user code a bit by instantiating the
  `ModelparamTraversal` inside the `multi_optimizer()` function.
- We keep the same behavior as in the previous implementation, where parameters
  are updated by the last optimizer in the sequence, in the case where a path
  matches more than one `FilterFn`.
- Note that the signature in `Optimizer` from the [functional API] needs to be
  changed slightly to accept and return
  `Union[OptimizerState, Sequence[OptimizerState]]` to make this work.

## Functional Usage
[Functional Usage]: #functional-usage

Using the [functional API] directly makes for the following code:

```python
def make_train_step(apply_fn, opt_update_fn):
  @jax.jit
  def train_step(opt_state, variables, inputs, labels):

    def loss_fn(params):
      logits, new_model_state = apply_fn(
          variables.copy(dict(params=params)), inputs, mutable=['batch_stats'])
      loss = xent_loss(logits, labels)
      return loss, new_model_state

    variables, params = variables.pop('params')
    (loss, new_model_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params)
    new_params, new_opt_state = opt_update_fn(
        grads, params, opt_state)
    new_variables = variables.copy(dict(params=new_params, **new_model_state))
    return new_opt_state, new_variables, loss

  return train_step

opt_state = optim.init(variables['params'])
train_step = make_train_step(model.apply, optim.update)
for batch in ds.as_numpy_iterator():
  opt_state, variables, loss = train_step(
      opt_state, variables, batch['image'], batch['label'])
```

Remarks:

- Compared with the [previous API], we can now keep the entire `variables`
  including the `params` as an input and output to the `train_step()`.
- Splitting `params` from `variables` is still necessary inside the train step
  because we only want to compute gradients with respect to `params` and not the
  entire `variables`.
- A further complication is that we cannot provide `apply_fn` and
  `opt_update_fn` as direct arguments to the jitted function `train_step()`
  because they're not valid JAX types - they need to be wrapped in a
  `make_train_step()` function like above, or declared as `static_argnums` to
  `jax.jit()`.

## Linen Helper
[Linen helper]: #linen-helper

While the above training loop is perfectly fine, a number of lines are going to
be exactly the same for every Linen training loop, namely:

1. The splitting of variables.
2. Passing in parameters separately because we only need the gradients for
   those.
3. Updating the parameters with the opimizer.
4. Merging the updated parameters, any other updated collections, and the
   unchanged collections to generate the new variables.
5. We also need to keep track of the model's `apply_fn`, the otpimizer's
   `update_fn`, and the optimizer state, in addition to the data.


If we put all this state into a dataclass then we only have to pass around a
single object (and the train data):

```python
@flax.struct.dataclass
class LinenHelper:
  apply_fn: Callable = flax.struct.field(pytree_node=False)
  variables: Dict[str, PyTree]
  optim: Optimizer = flax.struct.field(pytree_node=False)
  opt_state: OptimizerState
```

We can then augment this dataclass with some code (similar to
`flax.optim.Optimizer` and `flax.nn.Model`) that takes care of splitting and
merging the variable collections appropriately:

```python
  def update_with_loss(self, loss_fn, inputs, labels, *, mutable=False):

    def loss_from_params(params):
      outputs, new_model_state = self.apply_fn(
        self.variables.copy(dict(params=params)), inputs, mutable=mutable or [])
      loss = loss_fn(inputs, outputs, labels)
      return loss, new_model_state

    model_state, params = variables.pop('params')
    (loss, new_model_state), grads = jax.value_and_grad(
        loss_from_params, has_aux=True)(params)
    new_params, new_opt_state = self.optim.update(grads, params, self.opt_state)

    return self.replace(
        variables=variables.copy(dict(params=new_params, **new_model_state)),
        opt_state=new_opt_state,
    ), loss
```

Users can derive from this dataclass and put any additional state into that
object:

```python
@flax.struct.dataclass
class TrainState(LinenHelper):
  step: int
```

This will reduce the above training loop to something much simpler:

```python
@jax.jit
def train_step(state, inputs, labels):

  def loss_fn(inputs, outputs, labels):
    del inputs  # unused
    return xent_loss(outputs, labels)

  state, loss = state.replace(step=state.step + 1).update_with_loss(
      loss_fn, inputs, labels, mutable=['batch_stats'])
  return state, loss

opt_state = optim.init(variables['params'])
state = TrainState(
  apply_fn=model.apply,
  variables=variables,
  optim=optim,
  opt_state=opt_state,
  step=0,
)
for batch in ds.as_numpy_iterator():
  state, loss = train_step(state, batch['image'], batch['label'])
  print(loss)
```

Remarks:

- It is quite common to have `TrainState`-like objects in Flax that keep all
  the state (and some functions) and are used for checkpointing.
- Variables can be inspected and additionally modified in `train_step()` if
  needed.
- `LinenHelper` might provide more utility functions with slightly different
  signatures if there is demand for that.

# Previous API
[previous API]: #previous-api

## Previous Implementation
[Previous Implementation]: #previous-implementation

The optimizer itself would be implemented by creating a new class derived
from `OpimizerDef`:

```python
# flax/optim/momentum.py

@flax.struct.dataclass
class _MomentumHyperParams:
  learning_rate: jnp.ndarray
  beta: jnp.ndarray


@flax.struct.dataclass
class _MomentumParamState:
  momentum: np.ndarray


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

- Note the relationship between `OptimizerDef` and `Optimizer` : When the
  function `Optimizer.apply_gradient()` is called from the user code, it calls
  into `OptimizerDef.apply_gradient()` (among other things) which in turn will
  call `OptimizerDef.apply_param_gradient()` (implemented by subclasses of
  `OptimizerDef`).
- The functions `init_param_state()` and `apply_param_gradient()` are called
  for every leaf in the params/grads pytree. This makes it possible to write the
  calculations directly without `jax.tree_multimap()`.
- The interface was defined in pre-Linen without the distinction of `params` vs.
  other collections in `variables` in mind. The original API was elegant because
  one only needed to pass around the optimizer, which included the parameters,
  optimizer state, optimizer hyperparameters, and a reference to the
  `OptimizerDef` to perform the param/state update.

## Previous Usage
[Previous Usage]: #previous-usage

An optimizer would first be constructed from its definition and the pytree of
target params:

```python
optimizer_def = flax.optim.Momentum(learning_rate=0.1, beta=0.9)
optimizer = optimizer_def.create(variables['params'])
```

Then, the target variables would optimized in the train step (assuming a single
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

Remarks:

- Notice how `optimizer.apply_gradients()` can take additional arguments to
  update hyperparameters, such as learning rate from an independent function
  `get_learning_rate()` in this case.


# Update Plan
[Update Plan]: #update-plan

1. Finalize discussions on this FLIP
2. Test existing optimizers for numerical equivalence (e.g. `flax.optim.Adam`
   and `optax.adamw`).
3. Update examples to use Optax and verify that they reach the same final
   performance with the same computational cost. We probably want some examples
   to directly use the optimizer while others might use [Linen helper].
4. Port missing optimizers to Optax (e.g. Adafactor) - and verify above points.
5. Update all documentation (including README, Flax guided tour, HOWTOs, ...) to
   talk exclusively about Optax optimizers.
6. Mark optimizers in `flax.optim` as deprecated.

Note that all current Flax examples use an optimizer that is already available
in Optax:

| Example  |      Flax      |    Optax    |              Comments               |
| -------- | -------------- | ----------- | ----------------------------------- |
| imagenet | optim.Momentum | optax.sgd   | DynamicScale can be used unchanged. |
| mnist    | optim.Momentum | optax.sgd   |                                     |
| nlp_seq  | optim.Adam     | optax.adamw |                                     |
| pixelcnn | optim.Adam     | optax.adam  |                                     |
| ppo      | optim.Adam     | optax.adam  |                                     |
| seq2seq  | optim.Adam     | optax.adam  |                                     |
| vae      | optim.Adam     | optax.adam  |                                     |
| wmt      | optim.Adam     | optax.adamw |                                     |

(Flax's Adam implementation has an optional parameter for weight decay, but in
Optax Adam with and without weight decay are two different aliases.)