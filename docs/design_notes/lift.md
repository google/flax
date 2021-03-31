# Lifted transformation

⚠️ Advanced topic ⚠️

This design note explains the underlying implementation of `flax.linen.transform` which enable JAX transformations inside `Module`.


## Introduction

JAX uses a functional API meaning that it only garantuees correct behavior when using functions without side effects.
Typically these side effects are the result of mutating an object that lives outside of the function.

The functional paradigm has some advantages like the ability to reason about state and stochasticity explicitly.
A function output can only change when the input arguments change.
Therefore, a function is garantueed to behave deterministicly.

But pure functions offer another big advantage to JAX specifically: They enable functional transformations.
For example `jax.vmap(f)` will vectorize a function `f`.
Because `f` cannot have side-effects the vectorized/parallel version of `f` is well-defined. To see why we need this restriction, consider what happens if `f` would increment a counter or draw a random number.
Would `f` draw the same or a different random number for each item in the vector?
Would each item in the batch have its own counter or is the counter shared among the items?
And in what order is the counter incremented if `f` is computed in parallel?
The anser to all these questions is "it depends".
The behavior is ambigious and the functional constraint elegantly avoids this problem.

Flax introduces a safe way to have limited randomness and stateful variables in a JAX compatible form.
The reason state in Flax is not problematic is because it is local. Inside a flax `Module` there are variables and PRNG sequences,
but on the outside there are only JAX Arrays and PRNG keys.

For most use cases Flax be used to define models in a stateful way.
Becuase a `Module` behaves pure functional externally  we can utilize the full power of JAX with all of its transformations.
There are however cases where we want to have the best of both worlds by using transformations and Module together.
This design note explains how we extend JAX functional transformation to work on modules that have internal state and randomness.


## Functionalization

Before we jump into the details let's consider a simple example where we would like to use `vmap` inside a `Module`.

First we define a simple MLP without any transformations:
```
import jax
from jax import random, numpy as jnp
from flax import linen as nn

class MLP(nn.Module):
  @nn.compact
  def __call__(self, xs):
    h = nn.Dense(4, name='hidden')(xs)
    h = nn.relu(h)
    return nn.Dense(1, name='out')(h)
```

Now what if I want to have seperate MLP parameters for each item in `xs`?
If this where "vanilla JAX" we could imagene writing something like `jax.vmap(apply_mlp)(mlp_params, xs)`.
But doing something like this in linen will actually fail:
```
class NaiveVmapMLP(nn.Module):
  @nn.compact
  def __call__(self, xs):
    mlps = MLP()
    return jax.vmap(lambda mlp, x: mlp(x))(mlps, xs)  # fails
```

JAX will refuse to transform the `mlps` because its not a JAX Array or a simple container of arrays.
We can't really blame JAX for refusing this dirty, underspecified task.
After all it's not even clear what should happen here.
The parameters inside MLP are not even initialized yet and we will need a seperate PRNG key for each group of parameters.

We can fix this problem by first turning `MLP` into a pure init and apply function.
Afterwards, we use the `param` method to store the parameters:
```
class ManualVmap(nn.Module):
  @nn.compact
  def __call__(self, xs):
    mlp = MLP(parent=None)
    init_fn = lambda rng, xs: jax.vmap(mlp.init, in_axes=0)(random.split(rng, xs.shape[0]), xs)['params']
    apply_fn = jax.vmap(mlp.apply, in_axes=0)
    mlp_params = self.param('mlp', init_fn, xs)
    return apply_fn({'params': mlp_params}, xs)

xs = jnp.ones((3, 4))
variables = ManualVmap().init(random.PRNGKey(0), xs)
print(jax.tree_map(jnp.shape, variables['params']))
"""==>
{
    mlp: {
        hidden: {
            bias: (3, 4),
            kernel: (3, 4, 4),
        },
        out: {
            bias: (3, 1),
            kernel: (3, 4, 1),
        },
    },
}
"""
```

Here `MLP(parent=None)` creates a detached instance of `MLP`.
This avoids reserving a name for the submodule inside the current module.
Although not strictly necessary, this also ensures we cannot accidentaly use the MLP instance in a stateful way and we are forced to use it through either `.init` or `.apply`.

This example is still relatively concise but it already takes a few extra "bookkeeping" statements to make it work.
However, this implementation has a number of limitations:
1. During initialization we call the submodule twice through `init_fn` & `apply_fn`. If the submodule used the same trick to do
   functional tranformation we will end up executing a lot of code as the number of module calls grows like 2^d where d is the number of
   nested function transformations.
2. The implemenation assumes the submodule only requires the parameter RNG sequence.
3. The implementation assumes we init parameters once and we don't create any state variables either during init or apply.

Point 3 in particular makes manual functionalization cumbersome. Feel free to try and extend the above example with a `nn.BatchNorm`
layer in the `MLP` module. This should cause a number of complexites further growing the implementation. Like storing the updated
batch stats after apply and making sure state isn't mutable inside vmap when it's immutable outside (eval mode).


We call the process of transforming a stateful Module into a pure function "functionalization".
By temporarily turning a stateful Module into a function we make it compatible with JAX's functional transformations.

## Lifting


Flax provides an alternative for manual functionalization which we call lifted transformation.
Lifted transformations are defined in `flax.core.lift`.
All the lifted JAX transformations are defined with a single generic lifting API called `pack`.

A number of decision decisions had to be made in order to define `pack`. The implementation
of `pack` controls how variables and rngs are lifted and how fine-grained the user control is.
It must also decide whether lifting decisions are made during variable definition or at the transformation.


### Lifting granularity


With the linen API, users can define arbitrary variable collections and PRNG sequences.
Each variable in a collection is lifted in the same way.

Collections are typically given a semantically meaningful name like "params" or "batch_stats" rather than a general purpose name like "state".
Because collections carry semantic meaning we can decide at the transformation level how each collection should be lifted.
For example, we might decide to share parameters if we want to vectorize add a batch dimension to a model. 

At the same time we can write generic code that uses transformations without knowing exactly what kind of variables the submodules will create.
Collections thus strike a balance between fine-grained control and generality.
We also avoid brittle string matching code that loops over all variables and tries to split up collections in an ad-hoc way based on
naming conventions like: does this variable name have a prefix "kernel". 
If more fine-grained control is necessary a user can simply split up a set of variables over multiple collections that should be handled differently.


### Transformation vs variable control


Lifting behavior could be defined either at the transformation level or during variable definition.
We use transformation level definitions of lifting behavior.
The reason for this choice is that there are many different transformations with various behaviors.
For example: `vmap` has broadcasted and vectorized arguments, while `scan` has scan, carry, and broadcast arguments.
A variable would have to define its behavior for all these transformations otherwise a `Module` would not be compatible with
these tranformations. Or alternatively, we would have to make default decisions for how transformations are handled.
However, this could lead to silent bugs becuase the behavior might not actually be valid given the users intent. 

The lift package also provides a general purpose `transform` which allows an arbitrary function to transforms a variable collection.
This can be used to for example tie the weights in a tied autoencoder by transposing the weights.
It is unclear whether a similair general purpose transform could be defined if lifting decisions were made at variable definition.


### Linen

The lifting module does not know about linen Modules. Instead it operates directly on instances of `flax.core.Scope`.
A `Scope` contains the variables and PRNG sequences of a `Module`. Each module has a `Scope` instance linked to it as long
as its bound to a parent or it was created using a `init` or `apply`.

When a `Module` is transformed we use the `flax.core.lift` APIs to lift the scope and use `Module.clone()` to create a new `Module` instance with the lifted `Scope` bound to it.

`flax.linen.transforms` exposes wrappers for the transformations in `flax.core.lift`. The core lifting APIs operate on functions while
the linen wrappers can transform either a `Module` class or a `Module` method.

Lifting is thus implemented independently from the linen API. This seperation of concern simplifies the implementation while also allowing other Module APIs that chooce a different syntatic sugar for defining Modules to reuse the very generic and flexible `Scope `+ lift APIs for handling state and RNGs. 


### Implementation

The `pack(fn, in_vars, out_vars, rngs)` API goes through the following stages:


1. *Scope deduplication*

    This stage is only relevant if multiple Scopes are lifted together.
    In this case we must first find the set of root scopes.
    A scope is a root if none of its ancestors is in the set of scopes that is too be lifted.

    By only lifting roots we avoid lifting the same variables twice.

    For non-root scopes we store a reference to its ancestor scope and a path such that we can later reconstruct it (stage 4).

2. *Filter stage*

    Variables & PRNG sequences are split up into groups. This way `fn` can lift each group into the seperation seperatly.
    A group is defined by a filter which can be specified as:
    - a list of collections/prng names
    - `True` (match everything)
    - `False` (match nothing)
    - `DenyList(filter)` (match everything but the specified filter).

    A collection or PRNG sequence can only be put into a single group. If a collection matches multiple filters it will be put into
    the first group with a matching filter.
    For example, `in_vars = (["params"], True)` will cause the "params" collection to be put in the first group and all other collection to be put in the second group.

    For each PRNG sequence that is matched we seed a new PRNG sequence by calling `make_rng`.
    This avoids the need to update the PRNG state after the lifted tranformation finishes.

3. *Transform specific lifting*

    `fn` is called with the variable and PRNG groups.
    JAX transform have varying signatures and lifting options. Arguably the cleanest example is `vmap`.
    In the case of vmap the function arguments, rngs and variable collections are passed into a `jax.vmap` wrappped function.

4. *Scope reconstruction*

    Now that the variables and rngs are lifted inside the transformation, we want to recreate the lifted scopes. Pack calls
    `fn` with a `scope_fn` that takes the lifted variables and rngs and returns the reconstructed scopes with the lifted variables and rng sequences.

5. *Repack stage*

    After we have used the lifted scopes we have to retrieve the updated variables (PRNG sequences can simply be discarded).
    pack passes the `repack_fn` to support this.
    This stage is similair to stage 2 except that we only lift variables and immutable variables are ignored.
    Immutable variables cannot be updated so there is no reason to return them from the lifted transformation. 

6. *Commit stage*

    `pack` expects `fn` to return a pair where the first item will simply be returned from pack and the second item should be the repacked variables.
    The updated variables are stored in the original/unlifted scopes such that the mutation that happened inside the transformation survive beyond the transformation boundary.


### Using pack example


A minimal example of using pack:

```
from flax.core import lift
from flax.core import Scope, init, apply, nn as core_nn

def lift_id(fn, variables=True, rngs=True)
  def wrapper(scope_fn, repack_fn, variable_groups, rng_groups, *args):
    # normally we would first call into a JAX transform here...
    scope = scope_fn(variable_groups, rng_groups)
    y = fn(scope, *args)
    out_variables = repack_fn(scope)
    return y, out_variables

  return lift.pack(wrapper, (variables,), (variables,), (rngs,))

x = jnp.ones((3, 2))
y, params = init(lift_id(core_nn.dense))(random.PRNGKey(0), x, 4)
```

Normallly users shouldn't have to 

### Supported transformatiosn

| Jax Transform | Supported in Linen? | Comments |
|-|-|-|
| vmap | ✅ |  |
| scan | ✅ | Carry variables cannot be initialized inside the scan body. |
| remat | ✅ |  |
| jit | ✅ | Current implementation might cause unnecessary recompilation. |
| pmap | ❌ |  |
| grad | ❌ |  |
| xmap | ❌ |  |
| custom_vjp | ❌ |  |
| custom_jvp | ❌ |  |


### Linen examples

Going back to our original example we can now use the lifted vmap:

```
class LinenVmap(nn.Module):
  @nn.compact
  def __call__(self, xs):
    VmapMLP = nn.vmap(MLP, variable_axes={'params': 0}, split_rngs={'params': True}, in_axes=0)
    return VmapMLP(name='mlp')(xs)

variables = LinenVmap().init(random.PRNGKey(0), xs)
print(jax.tree_map(jnp.shape, variables['params']))
"""==>
{
    mlp: {
        Dense_0: {
            bias: (3, 4),
            kernel: (3, 2, 4),
        },
        Dense_1: {
            bias: (3, 1),
            kernel: (3, 4, 1),
        },
    },
}
"""
```

Here we use `variable_axes={'params': 0}` to indicate that paramaters are vectorized rather than shared and `split_rngs={'params': True}` means each set of parameters is initialized independently.

We can also extend the example with some inner state by adding a `BatchNorm` layer:

```
class StatefulMLP(nn.Module):
  @nn.compact
  def __call__(self, x):
    h = nn.Dense(4, name='hidden')(x)
    h = nn.BatchNorm(use_running_average=False, axis_name="batch")(h)
    h = nn.relu(h)
    return nn.Dense(1, name='out')(h)

class LinenStatefulVmap(nn.Module):
  @nn.compact
  def __call__(self, xs):
    VmapMLP = nn.vmap(StatefulMLP, variable_axes={'params': 0, 'batch_stats': 0}, split_rngs={'params': True}, in_axes=0)
    return VmapMLP(name='mlp')(xs)
variables = LinenStatefulVmap().init(random.PRNGKey(0), xs)
```

All we had to add to `nn.vmap` is `'batch_stats': 0` indicating the batch stats are vectorized rather than shared along the first axis.


## Alternatives

Other numerical computation frameworks consider variables a first-class citizen.
An alternative to functionalization would be to use a variable system either integrated or on top of JAX.
An advantage of this is that per-variable lifting becomes easier.
If variables are part of the JAX IR (JAXPR), we could inspect which variables have to be lifted in a certain computation.
Optionally, they could be annotated with a collection tag to decide on various lifting options.

The downside of this approach is that a variable system is more complicated.
Variables are related references and break a core assumption of Functional Programming (see [referentia transparency](https://en.wikipedia.org/wiki/Referential_transparency))
Other APIs that currently have a functional interface would probably require integration as well (e.g.: checkpointing and optimization APIs).