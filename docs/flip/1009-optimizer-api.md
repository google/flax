- Start Date: 2021-02-08
- FLIP PR: [#1011](https://github.com/google/flax/pull/1011)
- FLIP Issue: [#1009](https://github.com/google/flax/issues/1009)

Table of contents:

- [Summary]
- [Motivation]
- [Using Optax]
  - [Gradient Transformations]
  - [Optax Training Step]
  - [Multi Optimizer]
  - [Train State]
- [Previous API]
  - [Optimizer and OptimizerDef]
  - [Previous Training Step]
- [Update Plan]
- [Appendix]
  - [Setup Code]

# Summary
[Summary]: #summary

This FLIP proposes to replace our current `flax.optim` API (referred to as
[previous API] in this document) with [Optax], DeepMind's optimizer library.

# Motivation
[motivation]: #motivation

Our current API (referred to as [previous API] in this document) uses a pattern
where an `Optimizer` dataclass is created from a pytree of `target` variables
and from an `OptimizerDef` that defines how to update optimizer state,
hyperparameters, and target variables. This pattern is relatively complex for
implementing a simple optimizer, while being quite verbose in the typical Linen
train step (especially when using mutable state collections).

This package `flax.optim` contains some optimizers, but that list is far from
exhaustive and ideally we would instead use JAX optimizers from a dedicated PyPi
package.

DeepMind already has a dedicated library — [Optax] — that implements a wide
range of interesting optimizers and provides a framework to compose new
optimizers from reusable gradient transformations.

[Optax]: https://github.com/deepmind/optax

# Using Optax
[Using Optax]: #using-optax

## Gradient Transformations
[Gradient Transformations]: #gradient-transformations

While [Optax] does provide predefined optimizers (like `optax.adam`, or
`optax.sgd` with momentum), it is really a library of *gradient transformations*
and the idiomatic way of instantiating an optimizer is by providing a
combination of these gradient transformations. To emulate the momentum
optimizer from the example when using the [previous API] we would write:

```python
import optax

tx = optax.chain(
    optax.trace(decay=0.9, nesterov=False),
    optax.scale_by_schedule(lambda step: -get_learning_rate(step)),
)
```

Remarks:

- Above gradient transformation would be equivalent with the example under
  [Optimizer and OptimizerDef] where we define a Momentum optimizer without
  Nesterov momentum (note that the `beta` parameter corresponds to the `decay`
  parameter of the `optax.trace()` transformation, and the learning rate is
  applied in a second chained transformation).
- Note that hyper parameters like `decay` or `nesterov` only exist in the inner
  scope of the higher order functions returning the `GradientTransformation`.
  Such a gradient transformation is currently defined as a `NamedTuple` of the
  `init()` and the `update()` function. In principle this pattern could be
  extended to also store hyperparameters, maybe a point to discuss on the
  [Optax] repo.
- We can use a `get_learning_rate()` that returns the learning rate depending on
  the step number when defining the Optax gradient update transformation. Above
  code illustrates how this can be a drop-in replacement for a function we also
  use in our [previous training step], where this update function already exists
  (notice how we need to invert the sign because we add the gradient update to
  the parameters). In addition, you can use
  [`inject_hyperparams()`](https://github.com/deepmind/optax/pull/48) to
  schedule arbitrary hyper parameters with Optax.

## Optax Training Step
[Optax Training Step]: #optax-training-step

```python
@functools.partial(jax.jit, static_argnums=(4, 5))
def train_step(opt_state, variables, inputs, labels, apply_fn, tx_update_fn):

  def loss_fn(params):
    logits, new_model_state = apply_fn(
        {**variables, 'params': params}, inputs, mutable=['batch_stats'])
    loss = xent_loss(logits, labels)
    return loss, new_model_state

  variables, params = variables.pop('params')
  (loss, new_model_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
      params)
  updates, new_opt_state = tx_update_fn(grads, opt_state, params)
  new_params = optax.apply_updates(params, updates)
  new_variables = {**variables, **new_model_state, 'params': params}
  return new_opt_state, new_variables, loss


opt_state = tx.init(variables['params'])
for batch in ds.as_numpy_iterator():
  opt_state, variables, loss = train_step(
      opt_state, variables, batch['image'], batch['label'], model.apply,
      tx.update)
  print(loss)
```

Remarks:

- Since `tx.update()` only transforms the gradient, we still need to call
  `optax.apply_updates()` to apply these transformed gradients to the
  parameters.
- Compared with the [previous API], we can now keep the entire `variables`
  including the `params` as an input and output to the `train_step()`.
- Splitting `params` from `variables` is still necessary inside the train step
  because we only want to compute gradients with respect to `params` and not the
  entire `variables`.
- We can still log internal optimizer state, such as the learning rate, as long
  as Optax transformations expose that information in their respective state.
  For example, `optax.scale_by_schedule()` currently only exposes
  `opt_state.count` but could easily be extend to also expose the `step_size`.
  The same is true for internal optimizer states that change over time.

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

Optax has recently implemented `optax.masked()` that can be used for specifying
gradient transformations that only applied to a subset of the gradients:

```python
def flattened_traversal(fn):
  def mask(data):
    flat = traverse_util.flatten_dict(data)
    return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})
  return mask

tx = optax.chain(
    optax.masked(optax.sgd(learning_rate=0.1),
                 mask=flattened_traversal(lambda path, _: path[-1] == 'bias')),
    optax.masked(optax.sgd(learning_rate=0.05),
                 mask=flattened_traversal(lambda path, _: path[-1] != 'bias')),
)
```

## Train State
[Train State]: #train-state

In Flax it is common to hand around a `TrainState` object that can then be
used for checkpointing. This simplifies the above [Optax training step] a bit by
reducing the number of arguments and getting rid of the `static_argnums`.

We can define a `TrainState` dataclass that wraps the common pattern of updating
the optimizer state and parameters by applying the gradients.

```python
# Small helper class in flax.training
class TrainState(flax.struct.PyTreeNode):
  step: int
  apply_fn: Callable = flax.struct.field(pytree_node=False)
  params: flax.core.FrozenDict[str, Any]
  tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)
  opt_state: optax.OptState

  def apply_gradients(self, *, grads, **kwargs):
    updates, new_opt_state = self.tx.update(
        grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)
    return self.replace(
        step=self.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        **kwargs,
    )

  @classmethod
  def create(cls, *, apply_fn, params, tx, **kwargs):
    opt_state = tx.init(params)
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        tx=tx,
        opt_state=opt_state,
        **kwargs,
    )
```

Users can then derive from this dataclass and add more fields, for example
mutable model state:

```python
from flax.training import train_state

class TrainState(train_state.TrainState):
  batch_stats: flax.core.FrozenDict[str, Any]
```

With this the [Optax Training Step] becomes:

```python
@jax.jit
def train_step(state, inputs, labels):

  def loss_fn(params):
    outputs, new_model_state = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        inputs,
        mutable=['batch_stats'])
    loss = xent_loss(outputs, labels)
    return loss, new_model_state

  (loss, new_model_state), grads = jax.value_and_grad(
      loss_fn, has_aux=True)(state.params)
  new_state = state.apply_gradients(
      grads=grads,
      batch_stats=new_model_state['batch_stats'],
  )

  return new_state, loss


state = TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx,
    batch_stats=variables['batch_stats'],
)
for batch in ds.as_numpy_iterator():
  state, loss = train_step(state, batch['image'], batch['label'])
```

The train step without mutable state reduces to:

```python
@jax.jit
def train_step(state, inputs, labels):

  def loss_fn(params):
    outputs = state.apply_fn({'params': params}, inputs)
    loss = xent_loss(outputs, labels)
    return loss

  loss, grads = jax.value_and_grad(loss_fn)(state.params)
  new_state = state.update(grads=grads)

  return new_state, loss


state = flax.training.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx,
)
for batch in ds.as_numpy_iterator():
  state, loss = train_step(state, batch['image'], batch['label'])
```

Remarks:

- It is a common pattern in Flax training loops to have a `TrainState` dataclass
  that is updated with new state after every step.
- The simple solution proposed in `flax.training.train_state` an be extended
  with additional data, but advanced usecases (e.g. multiple different models
  and/or optimizers) are not supported. Users should instead fork the dataclass
  and re-implement it to their needs.
- As opposed to the `Optimizer` abstraction in the [previous API], the
  `TrainState` now directly contains the `.params`, without having to to through
  `.optimizer`

# Previous API
[previous API]: #previous-api

## Optimizer and OptimizerDef
[Optimizer and OptimizerDef]: #optimizer-and-optimizerdef

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
  calculations directly without `jax.tree_util.tree_map()`.
- The interface was defined in pre-Linen without the distinction of `params` vs.
  other collections in `variables` in mind. The original API was elegant because
  one only needed to pass around the optimizer, which included the parameters,
  optimizer state, optimizer hyperparameters, and a reference to the
  `OptimizerDef` to perform the param/state update.

## Previous Training Step
[Previous Training Step]: #previous-training-step

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

- Notice how `optimizer.apply_gradient()` can take additional arguments to
  update hyperparameters, such as learning rate from an independent function
  `get_learning_rate()` in this case.


# Update Plan
[Update Plan]: #update-plan

1. Finalize discussions on this FLIP
2. Add [equivalence tests] to Optax that guarantee that existing `flax.optim`
   optimizers return identical values with corresponding `optax` optimizers.
3. Update examples to use Optax and verify that they reach the same final
   performance with the same computational cost.
4. Port missing optimizers to Optax (e.g. Adafactor) - and verify above points.
5. Update all documentation (including README, Flax guided tour, HOWTOs, ...) to
   talk exclusively about Optax optimizers.
6. Create a transition guide for updating users from `flax.optim` to using
   Optax. This transition guide should also point to Optax's [equivalence tests]
   and the pull requests updating the examples.
7. Mark optimizers in `flax.optim` as deprecated.

[equivalence tests]: https://github.com/deepmind/optax/blob/master/optax/_src/equivalence_test.py

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

# Appendix
[Appendix]: #appendix

## Setup Code
[Setup Code]: #setup-code

The following setup code can be used for running the code snippets in this
FLIP:

```python
import functools
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import tensorflow as tf
import tensorflow_datasets as tfds


def pp(features):
  return {
      'image': tf.cast(features['image'], tf.float32) / 255 - 0.5,
      'label': features['label'],
  }


class Model(nn.Module):

  @nn.compact
  def __call__(self, inputs):
    x = inputs.reshape([inputs.shape[0], -1])
    x = nn.normalization.BatchNorm(True)(x)
    x = nn.Dense(10)(x)
    x = nn.log_softmax(x)
    return x


def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
  x = (labels[..., None] == jnp.arange(num_classes)[None])
  x = jax.lax.select(
      x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
  return x.astype(jnp.float32)


def xent_loss(logits, labels):
  return -jnp.sum(
      onehot(labels, num_classes=10) * logits) / labels.size


def get_learning_rate(step):
  return 0.1


model = Model()
rng = jax.random.PRNGKey(0)
ds = tfds.load('mnist')['train'].take(160).map(pp).batch(16)
batch = next(iter(ds))
variables = model.init(rng, jnp.array(batch['image'][:1]))
jax.tree_util.tree_map(jnp.shape, variables)
```
